import onmt
import torch.nn as nn
import torch
from torch.autograd import Variable


class Sampler(onmt.Translator):
  def __init__(self,opt):
    onmt.Translator.__init__(self, opt)
    self.samples = 1
    self.temperature = 1
    self.N = 5
    self.inp = torch.LongTensor(1,1).fill_(onmt.Constants.BOS)
    self.oneSentOut = torch.LongTensor(self.opt.max_sent_length,1)
    if self.opt.cuda:
          self.inp = self.inp.cuda()
          self.oneSentOut = self.oneSentOut.cuda()
    #self.inp = Variable(self.inp,volatile=True)

  def sample_one(self,src):
        srcBatch = Variable(torch.LongTensor([[self.src_dict.labelToIdx[x] if x in self.src_dict.labelToIdx else onmt.Constants.UNK for x in src[0]]]).t().contiguous())
        batchSize = 1
        #beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)
        #srcBatch = srcBatch[0] # drop the lengths needed for encoder

        rnnSize = context.size(2)
        decStates = (self.model._fix_enc_hidden(encStates[0]),
                      self.model._fix_enc_hidden(encStates[1]))

        decOut = self.model.make_init_decoder_output(context)

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        '''
        padMask = srcBatch.data.eq(onmt.Constants.PAD).t()
        def applyContextMask(m):
            if isinstance(m, onmt.modules.GlobalAttention):
                m.applyMask(padMask)
        '''

        #outputs = []
        for i in range(self.opt.max_sent_length):

          #self.model.decoder.apply(applyContextMask) 

          decOut, decStates, attn = self.model.decoder(
              Variable(self.inp,volatile=True), decStates, context, decOut)
          # decOut: 1 x batch x numWords
          decOut = decOut.squeeze(0)
          out = self.model.generator.forward(decOut)
          #outputs.append(out)
          word_weights = out.data.exp()
          self.inp = torch.multinomial(word_weights, 1)
          #_, inp = word_weights.max(1)
          self.oneSentOut[i] = self.inp

          #self.inp = self.inp.t().contiguous()
          
        return self.oneSentOut


  def sample(self,srcBatch, tgtBatch):
        dataset = self.buildData(srcBatch, tgtBatch)
        srcBatch, tgt, indices = dataset[0]
        batchSize = srcBatch[0].size(1)
        #beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)
        srcBatch = srcBatch[0] # drop the lengths needed for encoder

        rnnSize = context.size(2)
        decStates = (self.model._fix_enc_hidden(encStates[0]),
                      self.model._fix_enc_hidden(encStates[1]))

        decOut = self.model.make_init_decoder_output(context)

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = srcBatch.data.eq(onmt.Constants.PAD).t()
        def applyContextMask(m):
            if isinstance(m, onmt.modules.GlobalAttention):
                m.applyMask(padMask)

        inp = torch.LongTensor(batchSize,1).fill_(onmt.Constants.BOS).t().contiguous()
        sentOuts = torch.LongTensor(self.opt.max_sent_length,batchSize)
        if self.opt.cuda:
          inp = inp.cuda()
          sentOuts = sentOuts.cuda()
        outputs = []
        for i in range(self.opt.max_sent_length):

          self.model.decoder.apply(applyContextMask) 

          decOut, decStates, attn = self.model.decoder(
              Variable(inp, volatile=True), decStates, context, decOut)
          # decOut: 1 x batch x numWords
          decOut = decOut.squeeze(0)
          out = self.model.generator.forward(decOut)
          outputs.append(out)
          word_weights = out.data.exp()
          #inp = torch.multinomial(word_weights, 1)
          _, inp = word_weights.max(1)
          sentOuts[i] = inp

          inp = inp.t().contiguous()
          
        return outputs, sentOuts, dataset


