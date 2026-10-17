import struct, sys, collections
path = sys.argv[1] if len(sys.argv)>1 else 'Bonsai-8B.gguf'
f=open(path,'rb')
def rd(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
magic=f.read(4); ver=rd('<I')[0]; n_tensors=rd('<Q')[0]; n_kvs=rd('<Q')[0]
def read_string():
    n=rd('<Q')[0]; return f.read(n).decode('utf-8',errors='replace')
def read_value(t):
    if t==0: return rd('<B')[0]
    if t==1: return rd('<b')[0]
    if t==2: return rd('<H')[0]
    if t==3: return rd('<h')[0]
    if t==4: return rd('<I')[0]
    if t==5: return rd('<i')[0]
    if t==6: return rd('<f')[0]
    if t==7: return rd('<B')[0]
    if t==8: return read_string()
    if t==9:
        et=rd('<I')[0]; n=rd('<Q')[0]
        return [read_value(et) for _ in range(n)]
    if t==10: return rd('<Q')[0]
    if t==11: return rd('<q')[0]
    if t==12: return rd('<d')[0]
    raise Exception("unknown type %d"%t)
for i in range(n_kvs):
    k=read_string(); t=rd('<I')[0]; v=read_value(t)
ggml_types = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4",40:"NVFP4",41:"Q1_0"}
cnt=collections.Counter()
total_bits=0
for i in range(n_tensors):
    name=read_string()
    nd=rd('<I')[0]
    dims=[rd('<Q')[0] for _ in range(nd)]
    tt=rd('<I')[0]
    off=rd('<Q')[0]
    cnt[ggml_types.get(tt,str(tt))]+=1
    n=1
    for d in dims: n*=d
    print(f"{name:60s} {ggml_types.get(tt,tt):6s} {dims}")
print("TYPE COUNTS:", dict(cnt))
