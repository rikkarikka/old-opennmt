import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable


def train(train_iter, model, args, optimizer,steps):
    if args.cuda:
        model.cuda()


    model.train()
    for batch in train_iter:
        feature, target = batch[0], torch.LongTensor(batch[1])
        print(feature.size())
        #feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        optimizer.zero_grad()
        logit = model(feature)
        loss = F.cross_entropy(logit, Variable(target))
        loss.backward()
        optimizer.step()


    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    save_prefix = os.path.join(args.save_dir, 'snapshot')
    save_path = '{}_steps{}.pt'.format(save_prefix, steps)
    torch.save(model, save_path)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), feature.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = corrects/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))


def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]
