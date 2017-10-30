from __future__ import division

import onmt
import torch
import argparse
import math
from time import time

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=1,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=25,
                    help='Maximum sentence length.')
parser.add_argument('-no_unk', action="store_true",
                    help='No ukns in output')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")



def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def addone(f):
    for line in f:
        yield line
    yield None

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt)
    sampler = onmt.Sampler(opt)

    outF = open(opt.output, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    for line in addone(open(opt.src)):
        
        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]
        else:
            # at the end of file, exit 
            exit()

        start = time()
        sentOuts = sampler.sample_one(srcBatch) #
        sent = ""
        for i in sentOuts:
          if i[0]==onmt.Constants.EOS:
            break
          else: 
            sent += sampler.tgt_dict.idxToLabel[i[0]] +" "
        #predBatch, predScore, goldScore = translator.translate(srcBatch, tgtBatch)
        runtime = time() - start
        print(sent)
        print("RUN TIME: %.4f" % runtime)
        print('')

        srcBatch, tgtBatch = [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()


if __name__ == "__main__":
    main()
