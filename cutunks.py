import sys
import argparse

# usage: python3 cutunks.py -src <src> -target <tgt> -vocab <vocab> -t <threshold>
# this program cuts datapoint with too many unks in the 

parser = argparse.ArgumentParser(description='cutunks.py')

parser.add_argument('-src', required=True,
                     help="Path to the uncut src data")

parser.add_argument('-tgt', required=True,
                    help="Path to the uncut target data")

parser.add_argument('-src_vocab', required=True,
                    help="Path to src vocabulary")

parser.add_argument('-tgt_vocab', required=True,
                    help="Path to tgt vocabulary")

#parser.add_argument('-tgt_vocab', required=True,
#                    help="Path to tgt vocabulary")

parser.add_argument('-save', required=True,
                    help="Output prefix for the prepared data")

parser.add_argument('-t', type=int, default=2,
                    help="Threshold number of src UNKS")

parser.add_argument('-t2', type=int, default=2,
                    help="Threshold number of tgt UNKS")

opt = parser.parse_args()

def vocab(fn):
  with open(fn) as f:
    data = [x.split()[0] for x in f.readlines()]
  return set(data)

def line_count(fn):
  c = 0
  with open(fn) as f:
    for l in f:
      c+=1
  return c

if __name__=="__main__":
  srcv = vocab(opt.src_vocab)
  tgtv = vocab(opt.tgt_vocab)
  lines = line_count(opt.src)
  f = open(opt.src)
  g = open(opt.tgt)
  ff = open(opt.save+".src",'w')
  gg = open(opt.save+".tgt",'w')
  skip = 0
  ctr = 0
  for fl in f:
    fl = fl.lower()
    ctr += 1
    if ctr % 1000000 == 999999:
      print("Processed %d of %d lines, skipped %d"%(ctr,lines,skip))
    gl = next(g).lower()
    ok = True
    for s in fl.split(" . "):
      fls = set(s.split())
      if len(fls)-len(fls.intersection(srcv))>opt.t:
        ok = False
      
    #fls = set(fl.split())
    #gls = set(gl.split())
    #if len(fls)-len(fls.intersection(srcv))>opt.t:
    #  skip +=1
    #elif len(gls)-len(gls.intersection(tgtv))>opt.t2:
    #  skip +=1
    #else:
    if ok:
      for s in gl.split(" . "):
        gls = set(s.split())
        if len(gls)-len(gls.intersection(tgtv))>opt.t2:
          ok = False
          #print(s)
          #print(gls.difference(tgtv))
    if ok:
      ff.write(fl)
      gg.write(gl)
    else:
      skip +=1
  


