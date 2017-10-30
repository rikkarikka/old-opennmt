from __future__ import division

import onmt
import torch
import argparse
import math
import TrainDiscriminator as TD
from torch.autograd import Variable

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
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
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

# ARGS for discriminator
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max_norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel_num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')

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

    sampler = onmt.Sampler(opt)
    opt.embed_num = len(sampler.tgt_dict.idxToLabel)
    print(opt.embed_num)
    opt.class_num = 2
    discriminator = onmt.CNN_Text(opt)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    outF = open(opt.output, 'w')

    srcBatch, tgtBatch = [], []
    all_out, all_words, all_tgt = [],[],[]

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    for line in addone(open(opt.src)):
        
        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        outputs, words, dataset = sampler.sample(srcBatch, tgtBatch)
        srcB, tgtB, idxs = dataset[0]
        print(idxs)
        tgtB = tgtB.t()
        srcB, _ = srcB
        print(tgtB.size())
        words = words.t()
        all_out.append(outputs)
        all_words.append(words)
        ordered_tgt = torch.LongTensor(len(idxs),tgtB.size(1))
        print(ordered_tgt.size())
        j = 0
        for i in idxs:
          d = tgtB.data[i]
          print(d.size())
          ordered_tgt[j] = d
          j+=1
        all_tgt.append(ordered_tgt)
        for i in range(len(srcBatch)):
          j = idxs.index(i)
          print(srcBatch[i])
          s = words[j]
          w = sampler.buildTargetTokens(s,None,None)
          print(' '.join(w))
        srcBatch, tgtBatch = [], []

    #discriminate
    batches = [(Variable(all_words[i]),torch.LongTensor(all_words[i].size(0)).fill_(0)) for i in range(len(all_words))]
    batches += [(Variable(all_tgt[i]),torch.LongTensor(all_tgt[i].size(0)).fill_(1)) for i in range(len(all_words))]
    TD.train(batches,discriminator,opt,d_optimizer)



if __name__ == "__main__":
    main()
