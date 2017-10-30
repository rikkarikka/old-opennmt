import sys
import torch
import argparse

parser = argparse.ArgumentParser(description='translate.py')
parser.add_argument('-model',required=True,help="Path to model")
parser.add_argument('-lr',type=float,default=1.0,help="Learning Rate")
parser.add_argument('-save',required=True,help="Save model as")
opt = parser.parse_args()

m = torch.load(opt.model)
m['optim'].lr = opt.lr
torch.save(m,opt.save)
