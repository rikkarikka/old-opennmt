import sys
from nltk.translate import bleu_score

def bleu(f1,f2):
  with open(f1) as f:
    hyp = [x.split() for x in f.readlines()]
  
  with open(f2) as f:
    ref = [[x.split() for x in y.split(" | ")] for y in f.readlines()]
  
  assert(len(hyp)==len(ref))
  
  sco = bleu_score.corpus_bleu(ref,hyp)
  print("Corpus BLEU score: ",sco)

if __name__=="__main__":
  bleu(sys.argv[1],sys.argv[2])
