import struct, sys
path = sys.argv[1] if len(sys.argv)>1 else 'Bonsai-8B.gguf'
f=open(path,'rb')
def rd(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
magic=f.read(4); ver=rd('<I')[0]; n_tensors=rd('<Q')[0]; n_kvs=rd('<Q')[0]
print("magic",magic,"ver",ver,"ntensors",n_tensors,"nkvs",n_kvs)
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
    print(k, "=", v)
