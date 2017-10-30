import sys

with open(sys.argv[1]) as f:
  v1 = [x.split()[0] for x in f.readlines()]

with open(sys.argv[2]) as f:
  v2 = [x.strip().split()[0] for x in f.readlines()]

vlen = len(v2)
v2 = set(v2)

adders = []
for x in v1:
  if x not in v2:
    adders.append(x + " "+str(vlen))
    vlen += 1


with open(sys.argv[2],'a') as f:
  f.write("\n".join(adders))


