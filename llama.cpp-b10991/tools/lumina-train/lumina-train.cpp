// Lumina-Engine training tool: QLoRA fine-tuning of quantized GGUF models.
//
// Unlike the upstream `llama-finetune` example (which requires an F32 model), this
// tool trains low-rank adapters (LoRA) on top of a *frozen*, possibly quantized
// base model (Q4_K, Q8_0, Q1_0, ...). Because the base weights are never updated:
//
//   * memory stays small (no F32 copy of the model, no optimizer state for it),
//   * quantized storage is preserved (no dequantize -> train -> requantize round trip),
//   * the result is a standard LoRA adapter GGUF that llama.cpp can load directly.
//
// `.gguf.lumina` containers are transparently decompressed by the loader, so both
// `model.gguf` and `model.gguf.lumina` work as the base model.
//
// Dataset support (--dataset / --dataset-val):
//   * one JSON object per line, supporting three common shapes:
//       {"text": "..."}                                     continued pre-training (LM)
//       {"messages":     [{"role": "user", "content": "..."}, ...]}   OpenAI-style chat
//       {"conversations":[{"from": "human", "value": "..."}, ...]}    ShareGPT
//   * chat records are rendered with the model's own chat template (messages are
//     formatted incrementally so the assistant spans are known). With --mask-history
//     (default) only assistant turns contribute to the loss; the rest is masked.
//   * a separate validation file can be supplied with --dataset-val; otherwise the
//     validation set is carved out of the training stream using --val-split.
//
// Checkpoints (all optional):
//   * --save-every-steps N   save <out>.ckpt-<step>.gguf every N optimizer steps (default 500)
//   * --save-every-epochs N  save at the end of every N epochs (default 1)
//   * --keep-last N          keep only the last N checkpoints (default: keep all)
//   * --resume               restart from the newest checkpoint in the checkpoint dir

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

// Only tensors created by llama_adapter_lora_init_trainable() carry ".lora_" in
// their name; everything else (including the frozen base weights) is excluded.
static bool lumina_train_param_filter(const struct ggml_tensor * tensor, void * /*userdata*/) {
    return tensor != nullptr && tensor->name[0] != '\0' && strstr(tensor->name, ".lora_") != nullptr;
}

static std::vector<std::string> lumina_split_targets(const std::string & csv) {
    std::vector<std::string> out;

    size_t start = 0;
    while (start <= csv.size()) {
        const size_t comma = csv.find(',', start);
        const size_t end   = comma == std::string::npos ? csv.size() : comma;

        size_t b = start;
        size_t e = end;
        while (b < e && std::isspace((unsigned char) csv[b])) ++b;
        while (e > b && std::isspace((unsigned char) csv[e - 1])) --e;

        if (e > b) {
            out.emplace_back(csv.substr(b, e - b));
        }

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    return out;
}

//
// dataset: token stream + per-token trainability mask
//

struct lumina_token_stream {
    std::vector<llama_token> tokens;
    std::vector<uint8_t>     trainable; // 1 = token contributes to the loss, 0 = masked
    int64_t                  n_text = 0;
    int64_t                  n_chat = 0;
    int64_t                  n_bad  = 0;
};

// A read-only, pageable view of one token stream. A large `std::vector<llama_token>` is
// anonymous private memory: when the machine runs short of RAM the OS can only reclaim it
// by writing it to swap. Training the stage-1 corpus keeps ~170 M tokens (hundreds of MiB,
// and the vector capacity can be up to 2x that) live for the whole run. Backing the stream
// with a temp file and mapping it instead makes the pages clean and evictable, so a
// memory-constrained machine stays out of swap and simply re-faults them from the page
// cache. The data seen by the trainer is byte-for-byte identical.
struct lumina_stream_view {
    const int32_t * tokens   = nullptr;
    const uint8_t * mask     = nullptr;
    int64_t         n_tokens = 0;

    // mapping state (empty when the view points at an in-memory fallback)
    void *   map_base  = nullptr;
    size_t   map_bytes = 0;
    fs::path map_path;

#if defined(_WIN32)
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE map_handle  = nullptr;
#endif

    lumina_stream_view() = default;
    lumina_stream_view(const lumina_stream_view &) = delete;
    lumina_stream_view & operator=(const lumina_stream_view &) = delete;
    ~lumina_stream_view() { unmap(); }

    // point at the in-memory stream; used as a fallback when mapping is unavailable
    void use_memory(const lumina_token_stream & s) {
        tokens   = (const int32_t *) s.tokens.data();
        mask     = s.trainable.empty() ? nullptr : s.trainable.data();
        n_tokens = (int64_t) s.tokens.size();
    }

    bool is_mapped() const { return map_base != nullptr; }

    // Serialise the stream to `dir/tag.stream.bin` and map it read-only. On any failure the
    // view silently falls back to pointing at the in-memory vectors (return false).
    bool map_stream(const lumina_token_stream & s, const fs::path & dir, const std::string & tag) {
        use_memory(s);

        const size_t n = s.tokens.size();
        if (n == 0) {
            return false;
        }

        const size_t tok_bytes  = n * sizeof(int32_t);
        const size_t mask_bytes = s.trainable.size();
        const size_t mask_off   = (tok_bytes + 63) & ~(size_t) 63; // keep the mask aligned
        const size_t total      = mask_off + mask_bytes;

        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            return false;
        }

        const fs::path path = dir / (tag + ".stream.bin");
        map_path = path; // so a failure after this point is cleaned up by unmap()

        {
            std::ofstream of(path, std::ios::binary | std::ios::trunc);
            if (!of) {
                unmap();
                return false;
            }
            of.write((const char *) s.tokens.data(), (std::streamsize) tok_bytes);
            if (mask_bytes > 0) {
                static const char zeros[64] = { 0 };
                const size_t pad = mask_off - tok_bytes;
                if (pad > 0) {
                    of.write(zeros, (std::streamsize) pad);
                }
                of.write((const char *) s.trainable.data(), (std::streamsize) mask_bytes);
            }
            if (!of) {
                unmap();
                return false;
            }
        }

#if defined(_WIN32)
        file_handle = CreateFileW(path.wstring().c_str(), GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) {
            file_handle = INVALID_HANDLE_VALUE;
            unmap();
            return false;
        }
        map_handle = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (map_handle == nullptr) {
            CloseHandle(file_handle);
            file_handle = INVALID_HANDLE_VALUE;
            unmap();
            return false;
        }
        void * base = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);
        if (base == nullptr) {
            CloseHandle(map_handle);
            map_handle = nullptr;
            CloseHandle(file_handle);
            file_handle = INVALID_HANDLE_VALUE;
            unmap();
            return false;
        }
#else
        const int fd = open(path.string().c_str(), O_RDONLY);
        if (fd < 0) {
            unmap();
            return false;
        }
        void * base = mmap(nullptr, total, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (base == MAP_FAILED) {
            unmap();
            return false;
        }
#endif

        map_base  = base;
        map_bytes = total;
        tokens    = (const int32_t *) base;
        mask      = mask_bytes > 0
                  ? (const uint8_t *) ((const char *) base + mask_off)
                  : nullptr;

        return true;
    }

    void unmap() {
        if (map_base != nullptr) {
#if defined(_WIN32)
            UnmapViewOfFile(map_base);
#else
            munmap(map_base, map_bytes);
#endif
            map_base = nullptr;
        }
#if defined(_WIN32)
        if (map_handle != nullptr) {
            CloseHandle(map_handle);
            map_handle = nullptr;
        }
        if (file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle);
            file_handle = INVALID_HANDLE_VALUE;
        }
#endif
        if (!map_path.empty()) {
            std::error_code ec;
            fs::remove(map_path, ec);
            map_path.clear();
        }
    }
};

static void lumina_append(lumina_token_stream & out, const std::vector<llama_token> & toks, bool trainable) {
    out.tokens.insert(out.tokens.end(), toks.begin(), toks.end());
    out.trainable.insert(out.trainable.end(), toks.size(), (uint8_t) (trainable ? 1 : 0));
}

static bool lumina_load_dataset(
        llama_context *                          ctx,
        const llama_model *                      model,
        const common_params &                    params,
        const std::string &                      path,
        lumina_token_stream &                    out,
        common_chat_templates_ptr &              tmpls,
        bool &                                   tmpls_ready) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("%s: failed to open dataset '%s'\n", __func__, path.c_str());
        return false;
    }

    auto ensure_tmpls = [&]() -> bool {
        if (!tmpls_ready) {
            tmpls_ready = true;
            try {
                tmpls = common_chat_templates_init(model, params.dataset_chat_template);
            } catch (const std::exception & err) {
                LOG_ERR("%s: failed to initialise chat template: %s\n", __func__, err.what());
                return false;
            }
        }
        return tmpls != nullptr;
    };

    int64_t n_records = 0;

    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        const json j = json::parse(line, nullptr, false);
        if (j.is_discarded() || !j.is_object()) {
            ++out.n_bad;
            continue;
        }

        if (j.contains("text") && j["text"].is_string()) {
            const std::string text = j["text"].get<std::string>();
            const std::vector<llama_token> toks = common_tokenize(ctx, text, /*add_special=*/true, /*parse_special=*/false);
            lumina_append(out, toks, /*trainable=*/true);
            ++out.n_text;
        } else if ((j.contains("messages") && j["messages"].is_array()) ||
                   (j.contains("conversations") && j["conversations"].is_array())) {
            const bool   sharegpt = j.contains("conversations") && j["conversations"].is_array();
            const json & arr      = sharegpt ? j["conversations"] : j["messages"];

            std::vector<common_chat_msg> msgs;
            for (const auto & m : arr) {
                if (!m.is_object()) {
                    continue;
                }
                common_chat_msg msg;
                if (sharegpt) {
                    const std::string from = m.value("from", std::string());
                    msg.role    = (from == "human" || from == "user")      ? "user" :
                                  (from == "gpt"   || from == "assistant") ? "assistant" : from;
                    msg.content = m.value("value", std::string());
                } else {
                    msg.role    = m.value("role", std::string());
                    msg.content = m.value("content", std::string());
                }
                if (!msg.role.empty() && !msg.content.empty()) {
                    msgs.push_back(std::move(msg));
                }
            }

            if (msgs.empty()) {
                ++out.n_bad;
                continue;
            }

            if (!ensure_tmpls()) {
                return false;
            }

            // Format message-by-message so the assistant / non-assistant spans are known.
            std::vector<common_chat_msg> past;
            bool ok = true;
            for (const auto & msg : msgs) {
                std::string piece;
                try {
                    piece = common_chat_format_single(tmpls.get(), past, msg, /*add_ass=*/false, /*use_jinja=*/true);
                } catch (const std::exception & err) {
                    LOG_ERR("%s: chat template failed on '%s': %s\n", __func__, path.c_str(), err.what());
                    ok = false;
                    break;
                }
                const std::vector<llama_token> toks = common_tokenize(ctx, piece, /*add_special=*/false, /*parse_special=*/true);
                const bool trainable = !params.mask_history || msg.role == "assistant";
                lumina_append(out, toks, trainable);
                past.push_back(msg);
            }
            if (!ok) {
                return false;
            }
            ++out.n_chat;
        } else {
            ++out.n_bad;
            continue;
        }

        ++n_records;

        if (params.dataset_limit > 0 && n_records >= params.dataset_limit) {
            LOG_INF("%s: reached --dataset-limit (%d records)\n", __func__, params.dataset_limit);
            break;
        }
        if (params.dataset_max_tokens > 0 && (int64_t) out.tokens.size() >= params.dataset_max_tokens) {
            LOG_INF("%s: reached --dataset-max-tokens (%d tokens)\n", __func__, params.dataset_max_tokens);
            break;
        }
    }

    const size_t n_trainable = (size_t) std::count(out.trainable.begin(), out.trainable.end(), (uint8_t) 1);

    LOG_INF("%s: '%s': %" PRId64 " records (%" PRId64 " text, %" PRId64 " chat), %zu tokens (%zu trainable), %" PRId64 " skipped\n",
            __func__, path.c_str(), n_records, out.n_text, out.n_chat,
            out.tokens.size(), n_trainable, out.n_bad);

    return true;
}

// Pack one or two token streams into a ggml_opt_dataset. Labels are the next token
// (standard next-token prediction); masked positions get the sentinel label -1.
static ggml_opt_dataset_t lumina_dataset_init(
        llama_context *              ctx,
        const lumina_stream_view &   train,
        const lumina_stream_view *   val,
        float                        val_split,
        int64_t                      stride_opt,
        int64_t &                    idata_split) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    // Default (stride_opt == 0): overlap consecutive training windows by 50% (n_ctx/2).
    // A stride of n_ctx trains every token exactly once and halves the epoch time (each
    // token is otherwise the target of two different windows). It is user-selectable so the
    // existing training semantics stay the default; clamp to n_ctx so no token is skipped.
    const int64_t stride       = stride_opt > 0
                               ? std::min<int64_t>(stride_opt, ne_datapoint)
                               : std::max<int64_t>(1, ne_datapoint / 2);

    auto ndata_of = [&](int64_t n_tokens) -> int64_t {
        if (n_tokens <= ne_datapoint + 1) {
            return 0;
        }
        return (n_tokens - ne_datapoint - 1) / stride;
    };

    const int64_t ndata_train_all = ndata_of(train.n_tokens);

    int64_t ndata_train = 0;
    int64_t ndata_val   = 0;

    const bool have_val_file = val != nullptr && val->n_tokens > 0;

    if (have_val_file) {
        ndata_train = ndata_train_all;
        ndata_val   = ndata_of(val->n_tokens);
    } else {
        ndata_val   = (int64_t) (ndata_train_all * val_split);
        ndata_train = ndata_train_all - ndata_val;
    }

    if (ndata_train <= 0) {
        LOG_ERR("%s: not enough training data for context size %" PRId64 " (need > %" PRId64 " tokens per stream)\n",
                __func__, ne_datapoint, ne_datapoint + 1);
        return nullptr;
    }

    const int64_t ndata = ndata_train + ndata_val;

    // Compact backing: the previous implementation stored one full n_ctx copy of every
    // datapoint window plus its labels. With n_ctx=32768 and ~166M tokens that is about
    // 2.6 GiB of host memory that is mostly redundant (consecutive windows overlap by
    // half). Instead keep the token stream(s) once and materialise a window on demand.
    const int32_t * data_val  = have_val_file ? val->tokens : nullptr;
    const uint8_t * mask_main = train.mask;
    const uint8_t * mask_val  = have_val_file ? val->mask : nullptr;

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init_compact(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1,
        stride, ndata_train,
        train.tokens, mask_main,
        data_val, mask_val);

    idata_split = ndata_train;

    LOG_INF("%s: dataset: %" PRId64 " train datapoints, %" PRId64 " val datapoints, n_ctx = %" PRId64 ", stride = %" PRId64 " (compact)\n",
            __func__, ndata_train, ndata_val, ne_datapoint, stride);

    return dataset;
}

//
// checkpoints
//

struct lumina_ckpt_state {
    llama_adapter_lora * adapter = nullptr;
    std::string          stem;
    fs::path             dir;
    int32_t              save_every_steps  = 0;
    int32_t              save_every_epochs = 0;
    int32_t              keep_last         = 0;
    int64_t              cumulative_steps  = 0;
    uint32_t             epoch             = 0;
    uint32_t             epochs            = 0;
    int32_t              rank              = 0;
    float                alpha             = 0.0f;
    std::string          targets;
    std::string          base_model;
};

static lumina_ckpt_state g_ckpt;

static std::string lumina_checkpoint_prefix() {
    return g_ckpt.stem + ".ckpt-";
}

static void lumina_prune_checkpoints() {
    if (g_ckpt.keep_last <= 0) {
        return;
    }

    std::error_code ec;
    const std::string prefix = lumina_checkpoint_prefix();

    std::vector<std::pair<int64_t, std::string>> found; // step -> path without extension
    for (const auto & entry : fs::directory_iterator(g_ckpt.dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string fn = entry.path().filename().string();
        if (fn.rfind(prefix, 0) != 0 || entry.path().extension() != ".gguf") {
            continue;
        }
        std::string num = fn.substr(prefix.size());
        const size_t dot = num.find('.');
        if (dot != std::string::npos) {
            num = num.substr(0, dot);
        }
        try {
            found.emplace_back(std::stoll(num), entry.path().stem().string());
        } catch (...) {
            // ignore files that do not follow the naming convention
        }
    }

    std::sort(found.begin(), found.end());

    while ((int) found.size() > g_ckpt.keep_last) {
        const std::string stem = found.front().second;
        fs::remove(g_ckpt.dir / (stem + ".gguf"), ec);
        fs::remove(g_ckpt.dir / (stem + ".json"), ec);
        LOG_INF("%s: removed old checkpoint '%s'\n", __func__, stem.c_str());
        found.erase(found.begin());
    }
}

static void lumina_save_checkpoint(int64_t step, const char * reason) {
    if (g_ckpt.adapter == nullptr || g_ckpt.stem.empty()) {
        return;
    }

    std::error_code ec;
    fs::create_directories(g_ckpt.dir, ec);

    const std::string base = lumina_checkpoint_prefix() + std::to_string(step);
    const fs::path gguf_path = g_ckpt.dir / (base + ".gguf");
    const fs::path json_path = g_ckpt.dir / (base + ".json");

    if (!llama_adapter_lora_save(g_ckpt.adapter, gguf_path.string().c_str())) {
        LOG_ERR("%s: failed to save checkpoint '%s'\n", __func__, gguf_path.string().c_str());
        return;
    }

    json meta;
    meta["format"]       = "lumina-train-checkpoint";
    meta["version"]      = 1;
    meta["base_model"]   = g_ckpt.base_model;
    meta["step"]         = step;
    meta["epoch"]        = g_ckpt.epoch;
    meta["total_epochs"] = g_ckpt.epochs;
    meta["lora_rank"]    = g_ckpt.rank;
    meta["lora_alpha"]   = g_ckpt.alpha;
    meta["lora_targets"] = g_ckpt.targets;
    meta["reason"]       = reason != nullptr ? reason : "";

    std::ofstream jf(json_path, std::ios::binary);
    if (jf) {
        jf << meta.dump(2);
    }

    LOG_INF("%s: saved checkpoint '%s' (epoch %u/%u, step %" PRId64 ", %s)\n",
            __func__, gguf_path.string().c_str(), g_ckpt.epoch + 1, g_ckpt.epochs, step,
            reason != nullptr ? reason : "");

    lumina_prune_checkpoints();
}

// Returns the step of the newest checkpoint, or -1 if none was found.
static int64_t lumina_find_latest_checkpoint(uint32_t & epoch_out, std::string & gguf_out) {
    std::error_code ec;
    if (!fs::is_directory(g_ckpt.dir, ec)) {
        return -1;
    }

    const std::string prefix = lumina_checkpoint_prefix();

    std::vector<std::pair<int64_t, fs::path>> found;
    for (const auto & entry : fs::directory_iterator(g_ckpt.dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string fn = entry.path().filename().string();
        if (fn.rfind(prefix, 0) != 0 || entry.path().extension() != ".gguf") {
            continue;
        }
        std::string num = fn.substr(prefix.size());
        const size_t dot = num.find('.');
        if (dot != std::string::npos) {
            num = num.substr(0, dot);
        }
        try {
            found.emplace_back(std::stoll(num), entry.path());
        } catch (...) {
            // ignore
        }
    }

    if (found.empty()) {
        return -1;
    }

    std::sort(found.begin(), found.end());

    const auto & best = found.back();
    gguf_out = best.second.string();

    epoch_out = 0;
    fs::path json_path = best.second;
    json_path.replace_extension(".json");

    std::ifstream jf(json_path, std::ios::binary);
    if (jf) {
        const json meta = json::parse(jf, nullptr, false);
        if (!meta.is_discarded() && meta.contains("epoch") && meta["epoch"].is_number()) {
            epoch_out = (uint32_t) std::max<int64_t>(0, meta["epoch"].get<int64_t>());
        }
    }

    return best.first;
}

// Load an existing LoRA adapter (`path`) and copy its weights into the trainable
// adapter `adapter`. Used both for --init-lora (warm start) and for --resume.
static bool lumina_init_adapter_from_file(
        llama_model *        model,
        llama_adapter_lora * adapter,
        const std::string &  path,
        const char *         what) {
    LOG_INF("%s: loading %s '%s'\n", __func__, what, path.c_str());

    llama_adapter_lora * loaded = llama_adapter_lora_init(model, path.c_str());
    if (loaded == nullptr) {
        LOG_ERR("%s: failed to load %s '%s'\n", __func__, what, path.c_str());
        return false;
    }

    const bool ok = llama_adapter_lora_copy_weights(adapter, loaded);
    llama_adapter_lora_free(loaded);

    if (!ok) {
        LOG_ERR("%s: failed to copy weights from %s '%s' "
                "(the adapter rank and targets must match --lora-rank / --lora-targets)\n",
                __func__, what, path.c_str());
    }

    return ok;
}

static void lumina_train_callback(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    if (!train || g_ckpt.adapter == nullptr || g_ckpt.save_every_steps <= 0) {
        return;
    }

    ++g_ckpt.cumulative_steps;

    if (g_ckpt.cumulative_steps % g_ckpt.save_every_steps == 0) {
        lumina_save_checkpoint(g_ckpt.cumulative_steps, "step");
    }
}

// Drop the MoE expert FFN substrings from a comma separated LoRA target list, keeping the
// attention projections and the router. Expert weights are 3D [n_in, n_out, n_expert] and
// adapting every one of them with rank-64 AdamW needs several GiB of optimizer state,
// which does not fit on memory-constrained machines.
static std::string lumina_prune_expert_targets(const std::string & csv) {
    const std::vector<std::string> targets = lumina_split_targets(csv);

    std::vector<std::string> kept;
    bool have_router = false;
    bool removed     = false;

    for (const std::string & t : targets) {
        if (t.rfind("ffn", 0) == 0) {
            if (t == "ffn_gate_inp") {
                have_router = true;
                kept.push_back(t);
            } else {
                // broad FFN substrings (ffn_gate/ffn_up/ffn_down) match the huge MoE
                // expert tensors; drop them, keep the router and everything else
                removed = true;
            }
            continue;
        }
        kept.push_back(t);
    }

    if (!removed) {
        // nothing to prune: keep the caller's list verbatim
        return csv;
    }

    if (!have_router) {
        kept.push_back("ffn_gate_inp");
    }

    std::string out;
    for (size_t i = 0; i < kept.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += kept[i];
    }
    return out;
}

// Decide whether the requested LoRA targets fit in RAM together with the model, the KV
// cache and the (compact) dataset. If not, drop the MoE expert FFN targets (the dominant
// memory term) so training can proceed on low-memory machines.
static void lumina_fit_lora_targets(
        const llama_model * model,
        int64_t             n_ctx,
        size_t              dataset_tokens,
        common_params &     params) {
    if (params.lora_rank <= 0 || !common_model_has_experts(model)) {
        return;
    }

    auto adapter_bytes_of = [&](const std::string & csv) -> size_t {
        const std::vector<std::string> targets = lumina_split_targets(csv);
        std::vector<const char *> ptrs;
        ptrs.reserve(targets.size());
        for (const auto & t : targets) {
            ptrs.push_back(t.c_str());
        }
        return llama_adapter_lora_trainable_bytes(model, (uint32_t) params.lora_rank, ptrs.data(), ptrs.size());
    };

    const size_t adapter_bytes = adapter_bytes_of(params.lora_targets);
    if (adapter_bytes == 0) {
        return;
    }

    size_t free_ram  = 0;
    size_t total_ram = 0;
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev != nullptr) {
        ggml_backend_dev_memory(cpu_dev, &free_ram, &total_ram);
    }
    if (total_ram == 0) {
        return;
    }

    std::error_code ec;
    const size_t model_bytes = fs::file_size(params.model.path, ec);

    const int64_t n_layer   = llama_model_n_layer(model);
    const int64_t n_embd    = llama_model_n_embd(model);
    const int64_t n_head    = llama_model_n_head(model);
    const int64_t n_head_kv = llama_model_n_head_kv(model);
    const int64_t head_dim  = n_head > 0 ? n_embd / n_head : 0;
    const int64_t kv_heads  = n_head_kv > 0 ? n_head_kv : n_head;
    const size_t  kv_bytes  = (size_t) (2*n_layer*n_ctx*kv_heads*head_dim*sizeof(float));

    const size_t dataset_bytes = dataset_tokens*(sizeof(int32_t) + sizeof(uint8_t));
    const size_t opt_bytes     = adapter_bytes*4; // parameter + grad_acc + AdamW m/v
    const size_t overhead      = (size_t) 1536*1024*1024; // graph/activation scratch (rough)

    const size_t estimated = kv_bytes + dataset_bytes + opt_bytes + overhead;

    // Prune when the AdamW state alone would need more than ~45% of physical RAM. This
    // depends only on the model + rank + total memory, so stage1/2/3 always pick the same
    // target set (required for --init-lora), regardless of -c or dataset size.
    const size_t adapter_budget = (size_t) (0.45*(double) total_ram);

    LOG_INF("%s: LoRA weights %.2f GiB (AdamW keeps x4 = %.2f GiB, budget %.2f GiB); "
            "estimated private working set %.2f GiB of %.2f GiB RAM "
            "(model %.2f GiB is mmap'd and reclaimable)\n",
            __func__, adapter_bytes/1073741824.0, opt_bytes/1073741824.0, adapter_budget/1073741824.0,
            estimated/1073741824.0, total_ram/1073741824.0, model_bytes/1073741824.0);

    if (opt_bytes <= adapter_budget) {
        return;
    }

    const std::string pruned = lumina_prune_expert_targets(params.lora_targets);
    if (pruned == params.lora_targets) {
        LOG_WRN("%s: LoRA working set exceeds the available memory and no MoE expert targets "
                "could be pruned; training may be slow or fail\n", __func__);
        return;
    }

    const size_t pruned_bytes = adapter_bytes_of(pruned);

    LOG_WRN("%s: LoRA targets do not fit in RAM: dropping MoE expert FFN targets\n", __func__);
    LOG_WRN("%s:   '%s' -> '%s' (AdamW state %.2f -> %.2f GiB)\n", __func__,
            params.lora_targets.c_str(), pruned.c_str(),
            opt_bytes/1073741824.0, (pruned_bytes*4)/1073741824.0);
    LOG_WRN("%s:   pass --lora-targets explicitly on a machine with more RAM to keep expert adapters\n", __func__);

    params.lora_targets = pruned;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Training threads. --cpu-cores N pins the training to N threads; 0 (the default)
    // uses every hardware thread (logical cores). The generic engine default is the
    // number of *physical* cores, which leaves the SMT siblings of a 6c/12t CPU idle;
    // the backward kernels are memory-bandwidth bound, so the extra logical threads
    // are a real speed-up. This does not change the arithmetic: every kernel still
    // owns whole output rows, so the per-element accumulation order is identical
    // (verified bit-exact loss sequence vs. 6 threads).
    {
        int n_train_threads = params.cpu_cores > 0
                            ? params.cpu_cores
                            : (int) std::thread::hardware_concurrency();
        if (n_train_threads <= 0) {
            n_train_threads = 1;
        }
        params.cpuparams.n_threads       = n_train_threads;
        params.cpuparams_batch.n_threads = n_train_threads;
        LOG_INF("%s: training threads = %d (--cpu-cores %d)\n",
                __func__, n_train_threads, params.cpu_cores);
    }

    // KV cache must be F32 for training (no f16 backward pass for OUT_PROD)
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: forcing k cache type to f32 for training\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: forcing v cache type to f32 for training\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    const bool full_finetune = params.lora_rank <= 0;
    if (full_finetune) {
        // full fine-tuning requires writable F32 weights, exactly like upstream
        if (params.load_mode != LLAMA_LOAD_MODE_NONE) {
            LOG_INF("%s: full fine-tune: forcing load_mode = none to enable writable weights\n", __func__);
            params.load_mode = LLAMA_LOAD_MODE_NONE;
        }
    }

    // The backward pass needs GGML_OP_OUT_PROD with the (quantized) base weights as
    // source 0. Repacked/extra buffer types do not implement that op, so training must
    // use the native quantized buffers. This is a no-op for F32 models.
    if (!params.no_extra_bufts) {
        LOG_INF("%s: disabling weight repacking (required for the backward pass)\n", __func__);
        params.no_extra_bufts = true;
    }

    // Training backward kernels are CPU-only. When a GPU backend (e.g. Vulkan) is present the
    // graph scheduler would otherwise offload some forward ops, which corrupts the training
    // metrics (accuracy) and can change gradients. Force a pure-CPU training run.
    const bool allow_gpu_train = getenv("LUMINA_TRAIN_ALLOW_GPU") != nullptr;
    if (!allow_gpu_train) {
        if (params.n_gpu_layers != 0) {
            LOG_INF("%s: forcing n_gpu_layers = 0 (training backward pass is CPU-only)\n", __func__);
            params.n_gpu_layers = 0;
        }
        if (!params.no_op_offload) {
            LOG_INF("%s: disabling GPU op-offload (training backward pass is CPU-only)\n", __func__);
            params.no_op_offload = true;
        }
        // Restrict the model to the CPU device list. Keeping a GPU (e.g. Vulkan) device in
        // the list still makes its backend participate in graph buffer allocation, and on
        // an integrated GPU those buffers are carved out of the same system RAM the
        // training already needs -- which is what makes the tool fail with an out-of-memory
        // error even though training only ever runs on the CPU.
        params.devices = { nullptr };
        LOG_INF("%s: restricting device list to CPU only\n", __func__);
    } else {
        LOG_WRN("%s: LUMINA_TRAIN_ALLOW_GPU is set: GPU offload allowed for training "
                "(EXPERIMENTAL - only CPU implements the backward kernels, results may be wrong)\n", __func__);
    }
    // mark as explicit so the automatic device policy does not re-enable offload
    params.n_gpu_layers_set  = true;
    params.no_op_offload_set = true;

    if (params.out_file.empty()) {
        params.out_file = full_finetune ? "lumina-finetuned.gguf" : "lumina-lora.gguf";
    }

    if (!params.dataset.empty() && !params.dataset_val.empty()) {
        LOG_INF("%s: using a separate validation dataset: '%s'\n", __func__, params.dataset_val.c_str());
    } else if (!params.dataset_val.empty()) {
        LOG_ERR("%s: --dataset-val requires --dataset\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    // ------------------------------------------------------------------
    // build the token streams (dataset file(s) or a single --file/-p text)
    // ------------------------------------------------------------------
    const int64_t n_ctx_train = llama_n_ctx(ctx);
    lumina_token_stream train_stream;
    lumina_token_stream val_stream;
    const bool has_val_file = !params.dataset_val.empty();

    if (!params.dataset.empty()) {
        common_chat_templates_ptr tmpls;
        bool tmpls_ready = false;

        if (!lumina_load_dataset(ctx, model, params, params.dataset, train_stream, tmpls, tmpls_ready)) {
            return 1;
        }
        if (has_val_file) {
            if (!lumina_load_dataset(ctx, model, params, params.dataset_val, val_stream, tmpls, tmpls_ready)) {
                return 1;
            }
        }
    } else {
        if (params.prompt.empty()) {
            LOG_ERR("%s: no training data: pass --dataset <file.jsonl> or -f/-p <text>\n", __func__);
            return 1;
        }
        const std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, /*add_special=*/true);
        lumina_append(train_stream, toks, /*trainable=*/true);
        train_stream.n_text = 1;
        LOG_INF("%s: prompt tokens = %zu, n_ctx = %" PRId64 "\n", __func__, toks.size(), n_ctx_train);
    }

    if ((int64_t) train_stream.tokens.size() <= n_ctx_train + 1) {
        LOG_ERR("%s: training data is too short: %zu tokens, but the training context is %" PRId64 ".\n",
                __func__, train_stream.tokens.size(), n_ctx_train);
        LOG_ERR("%s: provide more data via --dataset, or reduce -c (the context is rounded up to a multiple of 256).\n", __func__);
        return 1;
    }

    int64_t idata_split = 0;

    // a NULL per-token mask means "all trainable" in the compact dataset; dropping an
    // all-ones mask saves one byte per token (e.g. ~160 MiB for the stage-1 corpus)
    auto drop_trivial_mask = [](lumina_token_stream & s) {
        if (!s.trainable.empty() &&
            std::all_of(s.trainable.begin(), s.trainable.end(), [](uint8_t v) { return v != 0; })) {
            std::vector<uint8_t>().swap(s.trainable);
        }
    };
    drop_trivial_mask(train_stream);
    drop_trivial_mask(val_stream);

    // Move the (potentially hundreds of MiB) token streams out of anonymous memory into
    // memory-mapped temp files. The trainer reads exactly the same bytes, but the pages are
    // now clean and file-backed, so the OS can reclaim them under pressure instead of
    // swapping the process out. When mapping is unavailable we keep the vectors (the view
    // then points at them) so the run still works.
    lumina_stream_view train_view;
    lumina_stream_view val_view;

    const fs::path stream_dir = fs::temp_directory_path() / "lumina-train";

    if (train_view.map_stream(train_stream, stream_dir, "train")) {
        std::vector<llama_token>().swap(train_stream.tokens);
        std::vector<uint8_t>().swap(train_stream.trainable);
        LOG_INF("%s: mapped training token stream to '%s' (%.1f MiB, evictable)\n",
                __func__, train_view.map_path.string().c_str(), train_view.map_bytes/1048576.0);
    } else {
        LOG_WRN("%s: could not map the training token stream; keeping it in RAM\n", __func__);
    }

    if (has_val_file) {
        if (val_view.map_stream(val_stream, stream_dir, "val")) {
            std::vector<llama_token>().swap(val_stream.tokens);
            std::vector<uint8_t>().swap(val_stream.trainable);
            LOG_INF("%s: mapped validation token stream to '%s' (%.1f MiB, evictable)\n",
                    __func__, val_view.map_path.string().c_str(), val_view.map_bytes/1048576.0);
        } else {
            LOG_WRN("%s: could not map the validation token stream; keeping it in RAM\n", __func__);
        }
    }

    ggml_opt_dataset_t dataset = lumina_dataset_init(
        ctx, train_view, has_val_file ? &val_view : nullptr, params.val_split,
        params.dataset_stride, idata_split);

    if (dataset == nullptr) {
        return 1;
    }

    // NOTE: the compact dataset references the views created above (either their mappings
    // or, as a fallback, the in-memory vectors), so those must stay alive until the dataset
    // is freed. They are released when main() returns, after training.

    struct lr_opt & lr = params.lr;

    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
    };

    // ------------------------------------------------------------------
    // checkpoints: locate the output stem and the checkpoint directory
    // ------------------------------------------------------------------
    const fs::path out_path(params.out_file);
    g_ckpt.stem = out_path.stem().string();
    g_ckpt.dir  = params.checkpoint_dir.empty()
                ? (out_path.has_parent_path() ? out_path.parent_path() : fs::path("."))
                : fs::path(params.checkpoint_dir);
    g_ckpt.save_every_steps  = params.save_every_steps;
    g_ckpt.save_every_epochs = params.save_every_epochs;
    g_ckpt.keep_last         = params.keep_last;
    g_ckpt.epochs            = lr.epochs;
    g_ckpt.rank              = params.lora_rank;
    g_ckpt.alpha             = params.lora_alpha;
    g_ckpt.targets           = params.lora_targets;
    g_ckpt.base_model        = params.model.path;

    // ------------------------------------------------------------------
    // trainable LoRA adapter (+ optional resume from a checkpoint)
    // ------------------------------------------------------------------
    llama_adapter_lora * adapter = nullptr;

    if (!full_finetune) {
        // reduce the LoRA target set if the MoE expert adapters would not fit in RAM
        lumina_fit_lora_targets(model, n_ctx_train,
                train_stream.tokens.size() + val_stream.tokens.size(), params);

        const std::vector<std::string> targets = lumina_split_targets(params.lora_targets);

        std::vector<const char *> target_ptrs;
        target_ptrs.reserve(targets.size());
        for (const auto & t : targets) {
            target_ptrs.push_back(t.c_str());
        }

        LOG_INF("%s: creating trainable LoRA adapter (rank = %d, alpha = %.3g, targets = %s)\n",
                __func__, params.lora_rank, (double) params.lora_alpha, params.lora_targets.c_str());

        adapter = llama_adapter_lora_init_trainable(
                model,
                (uint32_t) params.lora_rank,
                params.lora_alpha,
                target_ptrs.data(),
                target_ptrs.size());

        if (adapter == nullptr) {
            LOG_ERR("%s: failed to create trainable LoRA adapter\n", __func__);
            return 1;
        }

        float scale = 1.0f;
        llama_set_adapters_lora(ctx, &adapter, 1, &scale);

        lopt_params.param_filter    = lumina_train_param_filter;
        lopt_params.param_filter_ud = nullptr;
    } else {
        LOG_WRN("%s: full fine-tuning requested (-lora-rank 0): this only works for F32 models\n", __func__);
    }

    uint32_t start_epoch = 0;
    // position within `start_epoch` at which training resumes (0, 0 = start of epoch)
    int64_t  resume_idata_start  = 0;
    int64_t  resume_ubatch_start = 0;

    if (adapter != nullptr) {
        g_ckpt.adapter = adapter;

        // optional warm start: initialise the trainable adapter from an existing LoRA
        // (e.g. train stage-3 on top of a stage-2 adapter, then save stage-3 separately)
        if (!params.init_lora.empty()) {
            if (!lumina_init_adapter_from_file(model, adapter, params.init_lora, "initial LoRA")) {
                return 1;
            }
        }

        int64_t     ckpt_step  = -1;
        uint32_t    ckpt_epoch = 0;
        std::string ckpt_gguf;

        if (params.resume) {
            ckpt_step = lumina_find_latest_checkpoint(ckpt_epoch, ckpt_gguf);
        }

        if (params.resume && ckpt_step >= 0) {
            LOG_INF("%s: resuming from checkpoint '%s' (epoch %u, step %" PRId64 ")\n",
                    __func__, ckpt_gguf.c_str(), ckpt_epoch + 1, ckpt_step);

            llama_adapter_lora * loaded = llama_adapter_lora_init(model, ckpt_gguf.c_str());
            if (loaded == nullptr) {
                LOG_ERR("%s: failed to load checkpoint adapter, starting from scratch\n", __func__);
            } else {
                if (llama_adapter_lora_copy_weights(adapter, loaded)) {
                    g_ckpt.cumulative_steps = ckpt_step;
                    start_epoch             = std::min<uint32_t>(ckpt_epoch, lr.epochs);
                    LOG_INF("%s: restored LoRA weights, continuing at epoch %u\n", __func__, start_epoch + 1);

                    // The checkpoint step counts train micro-batches. Convert it back into
                    // (datapoint, micro-batch-within-datapoint) so the resumed epoch does not
                    // re-train the data that was already processed. The prefix of the resumed
                    // datapoint is still run forward (to rebuild its KV cache) but with all
                    // labels masked, so it contributes no gradient and no weight update.
                    const int64_t n_ctx_resume    = std::max<int64_t>(1, llama_n_ctx(ctx));
                    const int64_t n_batch_resume  = std::max<int64_t>(1, std::min<int64_t>(params.n_batch,  n_ctx_resume));
                    const int64_t n_ubatch_resume = std::max<int64_t>(1, std::min<int64_t>(params.n_ubatch, n_batch_resume));
                    const int64_t ubatch_per_ctx  = std::max<int64_t>(1, n_ctx_resume / n_ubatch_resume);
                    const int64_t steps_per_epoch = idata_split * ubatch_per_ctx;

                    if (steps_per_epoch > 0) {
                        const int64_t epoch_base = (int64_t) start_epoch * steps_per_epoch;
                        int64_t steps_in_epoch   = ckpt_step - epoch_base;
                        if (steps_in_epoch < 0 || steps_in_epoch > steps_per_epoch) {
                            // checkpoint written by a different configuration; restart the epoch
                            steps_in_epoch = 0;
                        }
                        resume_idata_start  = std::min<int64_t>(steps_in_epoch / ubatch_per_ctx, idata_split);
                        resume_ubatch_start = steps_in_epoch % ubatch_per_ctx;
                        if (resume_idata_start >= idata_split) {
                            resume_idata_start  = idata_split;
                            resume_ubatch_start = 0;
                        }
                        // keep the global micro-batch counter aligned with the resumed position
                        g_ckpt.cumulative_steps = epoch_base + steps_in_epoch;

                        LOG_INF("%s: resuming at datapoint %" PRId64 "/%" PRId64 " of epoch %u, "
                                "micro-batch %" PRId64 "/%" PRId64 " (global step %" PRId64 ")\n",
                                __func__, resume_idata_start, idata_split, start_epoch + 1,
                                resume_ubatch_start, ubatch_per_ctx, g_ckpt.cumulative_steps);
                    }
                } else {
                    LOG_ERR("%s: failed to copy checkpoint weights, starting from scratch\n", __func__);
                }
                llama_adapter_lora_free(loaded);
            }
        } else if (params.resume) {
            LOG_INF("%s: --resume given but no checkpoint found in '%s'\n", __func__, g_ckpt.dir.string().c_str());
        }

        if (!params.resume && !full_finetune) {
            LOG_INF("%s: checkpoints will be written to '%s' every %d step(s) / %d epoch(s)%s\n",
                    __func__, g_ckpt.dir.string().c_str(), g_ckpt.save_every_steps, g_ckpt.save_every_epochs,
                    g_ckpt.keep_last > 0 ? "" : " (keeping all)");
        }
    }

    llama_opt_init(ctx, model, lopt_params);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = start_epoch; lr.epoch < lr.epochs; ++lr.epoch) {
        g_ckpt.epoch = lr.epoch;
        LOG_INF("%s: epoch %u / %u\n", __func__, lr.epoch + 1, lr.epochs);

        const int64_t idata_start  = (lr.epoch == start_epoch) ? resume_idata_start  : 0;
        const int64_t ubatch_start = (lr.epoch == start_epoch) ? resume_ubatch_start : 0;
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split, idata_start, ubatch_start,
                        adapter != nullptr ? lumina_train_callback : ggml_opt_epoch_callback_progress_bar,
                        ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        if (adapter != nullptr && g_ckpt.save_every_epochs > 0 &&
            (((int) (lr.epoch + 1) % g_ckpt.save_every_epochs == 0) || lr.epoch + 1 == lr.epochs)) {
            lumina_save_checkpoint(g_ckpt.cumulative_steps, "epoch");
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    bool save_ok = false;
    if (adapter != nullptr) {
        save_ok = llama_adapter_lora_save(adapter, params.out_file.c_str());
        if (save_ok) {
            LOG_INF("%s: saved LoRA adapter to '%s'\n", __func__, params.out_file.c_str());
            LOG_INF("%s: load it with: -m <base-model> --lora '%s' (scale 1.0)\n", __func__, params.out_file.c_str());
        }
    } else {
        llama_model_save_to_file(model, params.out_file.c_str());
        save_ok = true;
        LOG_INF("%s: saved fine-tuned model to '%s'\n", __func__, params.out_file.c_str());
    }

    if (!save_ok) {
        LOG_ERR("%s: failed to save output to '%s'\n", __func__, params.out_file.c_str());
    }

    g_ckpt.adapter = nullptr;

    if (adapter != nullptr) {
        llama_adapter_lora_free(adapter);
    }

    llama_backend_free();

    return save_ok ? 0 : 1;
}
