import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t


cuda_if = settings.cuda_if

def mse(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda()
    return ((pre-rel)*mask).pow(2).mean()

def rmse(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda()
    return t.sqrt(((pre-rel)*mask).pow(2).mean())


def nmae(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda()
    def translate_mask(mask):
        u,v = t.where(mask == 1)
        return u,v
    u,v = translate_mask(1-mask)
    return t.abs(pre-rel)[u,v].mean()/(t.max(rel)-t.min(rel))

def mse_inv(pre,rel,mask=None):
    # loss = (rel-rel*pre*rel)\odot mask
    # rel \in R^{m\times n}, pre\in R{n\times m}
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda()
    pre_now = t.mm(t.mm(rel,pre),rel)
    rel_now = rel
    return mse(pre_now,rel_now,mask)

def mse_id(pre,rel,mask=None,direc='left'):
    # if direc == 'left': pre\in R^{m\times m}, rel \in R^{m\times n} calculate pre*rel-rel
    # else pre\in R^{n\times n}, rel \in R^{m\times n} calculate rel*pre-rel
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda()
    if direc == 'left':
        pre_now = t.mm(pre,rel)
    else:
        pre_now = t.mm(rel,pre)
    rel_now = rel
    return mse(pre_now,rel_now,mask)



