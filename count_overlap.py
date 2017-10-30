import sys
import os
from time import time

swdict = {}
with open(sys.argv[1]) as f:
  for l in f:
    l = l.split()[0]
    swdict[l.strip()] = 0

i = 0
with open(sys.argv[2]) as f:
  for l in f:
    i+=1
    if i%100000==99999:
      print(i)
      oov = len(swdict)-len([0 for k,v in swdict.items() if v>1])
      print("oov @ 1: ",oov)
      ooov = len(swdict)-len([0 for k,v in swdict.items() if v>5])
      print("oov @ 5: ",ooov)
    wrds = l.strip().split()
    for x in wrds:
      if x in swdict:
        swdict[x]+=1


