from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import onmt


class Dataset(object):

    def __init__(self, srcData, tgtData, batchSize, cuda, volatile=False, byLengths=True):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.volatile = volatile
        self.byLengths = byLengths

        self.starts = []
        curr_length = -1

        
        for i in range(0, len(self.src)):
            if len(self.src[i]) != curr_length or i - self.starts[-1] >= self.batchSize:
                curr_length = len(self.src[i])
                self.starts.append(i)

        self.numBatches = len(self.starts)


    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        if self.byLengths:
            start = self.starts[index]
            end = self.starts[index + 1] if index != len(self.starts) - 1 else len(self.src)
        else:
            start = index * self.batchSize
            end = (index + 1) * self.batchSize
        
        srcBatch, lengths = self._batchify(
            self.src[start:end],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[start:end])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch = zip(*batch)
        else:
            indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)

        return (wrap(srcBatch), lengths), wrap(tgtBatch), indices

    def __len__(self):
        return self.numBatches


    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
