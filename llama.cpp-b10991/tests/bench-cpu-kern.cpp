// Micro-benchmark for the repacked CPU Q4_K GEMV kernel.
//
// Why: token generation on CPU is dominated by the quantized GEMV kernels, and
// end-to-end benchmarks (llama-bench) are far too noisy to resolve a few percent
// on a busy machine. This harness generates its own Q4_K data, calls the *real*
// kernel entry points from ggml/src/ggml-cpu/repack.h, verifies the result against
// an independent scalar reference, and A/B tests experimental kernel variants
// against the shipped one **inside one process, interleaved**, so CPU frequency /
// background-load drift cancels out.
//
// The experimental variants are generated from the real source text by
// bench/gen_q4k_body.py (see tests/q4k_gemv_body.inc) so they are byte-identical
// to the shipped kernel apart from the injected change.
//
// usage: bench-cpu-kern [--n K] [--nc N] [--iters I] [--rounds R] [--check-only]

#define GGML_COMMON_DECL_CPP

#include "ggml.h"
#include "ggml-common.h"

#include "ggml-cpu/repack.h"

#include <immintrin.h>

#define UNUSED GGML_UNUSED

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>

// ---------------------------------------------------------------------------
// experimental variants of the shipped kernel, generated from the real source
// ---------------------------------------------------------------------------

#define GEMV_FN gemv_q4_K_copy
#define GEMV_PDIST 0
#define GEMV_GUARD 0
#include "q4k_gemv_body.inc"
#undef GEMV_FN
#undef GEMV_PDIST
#undef GEMV_GUARD

#define GEMV_FN gemv_q4_K_pf_full_1
#define GEMV_PDIST 1
#define GEMV_GUARD 0
#include "q4k_gemv_body.inc"
#undef GEMV_FN
#undef GEMV_PDIST
#undef GEMV_GUARD

#define GEMV_FN gemv_q4_K_pf_full_2
#define GEMV_PDIST 2
#define GEMV_GUARD 0
#include "q4k_gemv_body.inc"
#undef GEMV_FN
#undef GEMV_PDIST
#undef GEMV_GUARD

#define GEMV_FN gemv_q4_K_pf_full_4
#define GEMV_PDIST 4
#define GEMV_GUARD 0
#include "q4k_gemv_body.inc"
#undef GEMV_FN
#undef GEMV_PDIST
#undef GEMV_GUARD

#define GEMV_FN gemv_q4_K_pf_full_2g
#define GEMV_PDIST 2
#define GEMV_GUARD 1
#include "q4k_gemv_body.inc"
#undef GEMV_FN
#undef GEMV_PDIST
#undef GEMV_GUARD

#define GEMV_FN gemv_q4_K_pf_head_2
#define GEMV_PDIST -2
#define GEMV_GUARD 0
#include "q4k_gemv_body.inc"
#undef GEMV_FN
#undef GEMV_PDIST
#undef GEMV_GUARD

// ---------------------------------------------------------------------------

typedef void (*gemv_fn)(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);

static uint32_t g_rng = 0x12345678u;
static float frand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float) ((int32_t) (g_rng >> 9) - (1 << 22)) / (float) (1 << 22);
}

static double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// independent scalar reference
// ---------------------------------------------------------------------------

static inline void get_scale_min_k4_ref(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static double ref_dot_q4_K_q8_K(const block_q4_K * x, const block_q8_K * y) {
    const double d    = ggml_fp16_to_fp32(x->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
    const double dmin = ggml_fp16_to_fp32(x->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);

    double sum = 0.0;
    for (int j = 0; j < QK_K / 32; ++j) {
        uint8_t sc = 0, m = 0;
        get_scale_min_k4_ref(j, x->scales, &sc, &m);
        const double dl = d * sc;
        const double ml = dmin * m;
        for (int i = 0; i < 32; ++i) {
            const uint8_t packed = x->qs[32 * (j / 2) + i];
            const int q = (j & 1) ? (packed >> 4) : (packed & 0x0F);
            const double w = dl * (double) q - ml;
            sum += w * (double) y->d * (double) y->qs[j * 32 + i];
        }
    }
    return sum;
}

static void quantize_q8_K(const float * x, block_q8_K * y) {
    float amax = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }
    const float iscale = amax > 0.0f ? -127.0f / amax : 0.0f;
    y->d = iscale != 0.0f ? 1.0f / iscale : 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        y->qs[i] = (int8_t) fminf(127.0f, fmaxf(-127.0f, roundf(x[i] * iscale)));
    }
    for (int i = 0; i < QK_K / 16; ++i) {
        int sum = 0;
        for (int j = 0; j < 16; ++j) {
            sum += y->qs[i * 16 + j];
        }
        y->bsums[i] = (int16_t) sum;
    }
}

// identical to make_block_q4_Kx8() in ggml/src/ggml-cpu/repack.cpp
static block_q4_Kx8 make_block_q4_Kx8(const block_q4_K * in, unsigned blck_size_interleave) {
    block_q4_Kx8 out;
    for (int i = 0; i < 8; i++) {
        out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 4 / (int) blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        const int src_id     = i % 8;
        const int src_offset = (i / 8) * (int) blck_size_interleave;
        const int dst_offset = i * (int) blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }

    uint8_t s[8], m[8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }
        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
        }
        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    return out;
}

// single threaded streaming-read ceiling of the machine, for calibration
static double stream_read_gbps(size_t bytes, int iters) {
    const size_t n   = bytes / sizeof(float);
    float *      buf = (float *) malloc(bytes);
    for (size_t i = 0; i < n; i++) {
        buf[i] = 1.0f;
    }
    volatile float sink = 0.0f;
    for (size_t i = 0; i < n; i += 8) {
        sink += buf[i];
    }
    const double t0 = now_s();
    for (int it = 0; it < iters; it++) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; i += 8) {
            acc += buf[i];
        }
        sink += acc;
    }
    const double t1 = now_s();
    free(buf);
    (void) sink;
    return (double) bytes * iters / (t1 - t0) / 1e9;
}

int main(int argc, char ** argv) {
    int64_t n          = 4096;
    int64_t nc         = 4096;
    int     iters      = 100;
    int     rounds     = 7;
    bool    check_only = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc)          { n = atoll(argv[++i]); }
        else if (!strcmp(argv[i], "--nc") && i + 1 < argc)    { nc = atoll(argv[++i]); }
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) { iters = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) { rounds = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--check-only"))            { check_only = true; }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }

    if (n % QK_K != 0 || nc % 8 != 0) {
        fprintf(stderr, "n must be a multiple of %d and nc a multiple of 8\n", QK_K);
        return 1;
    }

    const int64_t nb = n / QK_K;

    std::vector<float> wf((size_t) n * nc);
    for (auto & v : wf) {
        v = 0.05f * frand();
    }
    std::vector<float> act((size_t) n);
    for (auto & v : act) {
        v = frand();
    }

    std::vector<block_q4_K> wq((size_t) nb * nc);
    const size_t qsize = ggml_quantize_chunk(GGML_TYPE_Q4_K, wf.data(), wq.data(), 0, nc, n, nullptr);
    if (qsize != (size_t) nb * nc * sizeof(block_q4_K)) {
        fprintf(stderr, "unexpected quantized size %zu\n", qsize);
        return 1;
    }

    std::vector<block_q4_Kx8> wrep((size_t) nb * nc / 8);
    for (int64_t c = 0; c < nc; c += 8) {
        for (int64_t b = 0; b < nb; b++) {
            block_q4_K rows[8];
            for (int r = 0; r < 8; r++) {
                rows[r] = wq[(size_t) (c + r) * nb + b];
            }
            wrep[(size_t) (c / 8) * nb + b] = make_block_q4_Kx8(rows, 8);
        }
    }

    std::vector<block_q8_K> aq(nb);
    for (int64_t b = 0; b < nb; b++) {
        quantize_q8_K(act.data() + b * QK_K, &aq[b]);
    }

    const size_t bytes = (size_t) nc * nb * sizeof(block_q4_K);

    struct variant {
        const char * name;
        gemv_fn      fn;
    };

    const variant variants[] = {
        { "shipped        ", ggml_gemv_q4_K_8x8_q8_K },
        { "copy(pdist=0)  ", gemv_q4_K_copy        },
        { "prefetch full 1", gemv_q4_K_pf_full_1    },
        { "prefetch full 2", gemv_q4_K_pf_full_2    },
        { "prefetch full 4", gemv_q4_K_pf_full_4    },
        { "prefetch f2+guard", gemv_q4_K_pf_full_2g },
        { "prefetch head 2", gemv_q4_K_pf_head_2    },
    };
    const int nvar = (int) (sizeof(variants) / sizeof(variants[0]));

    // ---- correctness -------------------------------------------------------
    std::vector<float> ref(nc, 0.0f);
    for (int64_t c = 0; c < nc; c++) {
        double acc = 0.0;
        for (int64_t b = 0; b < nb; b++) {
            acc += ref_dot_q4_K_q8_K(&wq[(size_t) c * nb + b], &aq[b]);
        }
        ref[c] = (float) acc;
    }

    std::vector<float> out(nc, 0.0f);
    bool fail = false;
    for (int v = 0; v < nvar; v++) {
        std::fill(out.begin(), out.end(), 0.0f);
        variants[v].fn((int) n, out.data(), 0, wrep.data(), aq.data(), 1, (int) nc);

        double max_abs = 0.0, ref_scale = 0.0;
        for (int64_t c = 0; c < nc; c++) {
            max_abs   = fmax(max_abs, fabs((double) out[c] - (double) ref[c]));
            ref_scale = fmax(ref_scale, fabs((double) ref[c]));
        }
        const double tol = 1e-6 * (ref_scale > 0.0 ? ref_scale : 1.0) + 1e-7;

        std::vector<float> out_shipped(nc, 0.0f);
        ggml_gemv_q4_K_8x8_q8_K((int) n, out_shipped.data(), 0, wrep.data(), aq.data(), 1, (int) nc);
        const bool identical = memcmp(out.data(), out_shipped.data(), nc * sizeof(float)) == 0;

        printf("%s : vs scalar ref max_abs_err=%.3e (tol %.3e) %s ; bit-identical to shipped: %s\n",
               variants[v].name, max_abs, tol, max_abs <= tol ? "OK" : "FAIL",
               identical ? "yes" : "NO");
        fail |= (max_abs > tol);
    }
    if (fail) {
        return 2;
    }
    if (check_only) {
        return 0;
    }

    // ---- throughput: interleaved A/B ---------------------------------------
    printf("\nstreaming read ceiling (single thread, 32 MiB): %.2f GB/s\n",
           stream_read_gbps(32u << 20, 20));
    printf("matrix: n=%lld nc=%lld (%.2f MiB), iters=%d rounds=%d\n\n",
           (long long) n, (long long) nc, (double) bytes / (1024.0 * 1024.0), iters, rounds);

    std::vector<std::vector<double>> samples(nvar, std::vector<double>(rounds, 0.0));
    for (int r = 0; r < rounds; r++) {
        // rotate the order so that no variant is systematically measured first
        for (int k = 0; k < nvar; k++) {
            const int v = (r + k) % nvar;
            const double t0 = now_s();
            for (int it = 0; it < iters; it++) {
                variants[v].fn((int) n, out.data(), 0, wrep.data(), aq.data(), 1, (int) nc);
            }
            const double t1 = now_s();
            samples[v][r] = (double) bytes * iters / (t1 - t0) / 1e9;
        }
    }

    for (int v = 0; v < nvar; v++) {
        std::vector<double> s = samples[v];
        std::sort(s.begin(), s.end());
        const double med = s[s.size() / 2];
        const double bw  = med;
        // per-round ratio against the shipped kernel: robust against common-mode noise
        // (CPU frequency drift, background load bursts)
        std::vector<double> ratio(rounds);
        for (int r = 0; r < rounds; r++) {
            ratio[r] = samples[v][r] / samples[0][r];
        }
        std::sort(ratio.begin(), ratio.end());
        // "best quarter" mean: the rate the kernel reaches when nothing else steals the bus
        const int nq = std::max(1, rounds / 4);
        double bq = 0.0;
        for (int i = 0; i < nq; i++) {
            bq += s[rounds - 1 - i];
        }
        bq /= nq;
        printf("%s : median %6.2f  best25%% %6.2f  max %6.2f GB/s  %5.2f Gweights/s  | per-round vs shipped: %+.1f%% (p25 %+.1f%%, p75 %+.1f%%)\n",
               variants[v].name, med, bq, s.back(), bw * (double) QK_K / (double) sizeof(block_q4_K),
               100.0 * (ratio[rounds / 2] - 1.0), 100.0 * (ratio[rounds / 4] - 1.0),
               100.0 * (ratio[3 * rounds / 4] - 1.0));
    }

    return 0;
}
