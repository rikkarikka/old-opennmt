import onmt

import argparse
import torch

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                     help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")


parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-tgt_seq_length', type=int, default=125,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

parser.add_argument('-cut_unks', type=int, default=0,
                    help="Cut sentences with more than this many <unks>")

parser.add_argument('-vocabs', action='store_true', help='just make vocabs not data')

parser.add_argument('-oov_unk', type=int, default=0,
                    help="tokens appearing less frequently are unked")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD], lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    if opt.oov_unk:
        vocab = vocab.cut(opt.oov_unk)
    else:
        vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0
    unk_ignored = 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcWords = sline.split()
        tgtWords = tline.split()

        if len(srcWords) <= opt.seq_length and len(tgtWords) <= opt.tgt_seq_length:

            s = [srcDicts.convertToIdx(srcWords,
                                          onmt.Constants.UNK_WORD)]
            t = [tgtDicts.convertToIdx(tgtWords,
                                        onmt.Constants.UNK_WORD,
                                        onmt.Constants.BOS_WORD,
                                        onmt.Constants.EOS_WORD)]

            if opt.cut_unks == 0:
              ok = True
            else:
              sunk = s[0].eq(srcDicts.lookup(onmt.Constants.UNK_WORD)).sum()
              tunk = t[0].eq(tgtDicts.lookup(onmt.Constants.UNK_WORD)).sum()
              if sunk <= opt.cut_unks or tunk <= opt.cut_unks:
                ok = True
              else:
                ok = False
            if ok:
              src += s
              tgt += t
              sizes += [len(srcWords)]
            else:
              unk_ignored+=1
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    print(' %d ignored due to unks > %d' % (unk_ignored, opt.cut_unks))
    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length' % # == 0 or > %d)' %
          (len(src), ignored)) #, opt.seq_length))
    print(' %d ignored due to unks > %d' % (unk_ignored, opt.cut_unks))

    return src, tgt


def main():

    dicts = {}
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size)
    dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size)
    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    if not opt.vocabs:
      print('Preparing training ...')
      train = {}
      train['src'], train['tgt'] = makeData(opt.train_src, opt.train_tgt,
                                            dicts['src'], dicts['tgt'])

      print('Preparing validation ...')
      valid = {}
      valid['src'], valid['tgt'] = makeData(opt.valid_src, opt.valid_tgt,
                                      dicts['src'], dicts['tgt'])



      print('Saving data to \'' + opt.save_data + '.train.pt\'...')
      save_data = {'dicts': dicts,
                   'train': train,
                   'valid': valid}
      torch.save(save_data, opt.save_data + '.train.pt')

if __name__ == "__main__":
    main()
