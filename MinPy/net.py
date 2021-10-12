import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t
from torch.autograd import Variable

from third_party.models import *
from third_party.utils.denoising_utils import *

cuda_if = settings.cuda_if


class dmf(object):
    # Deep Matrix Factorization
    def __init__(self,params):
        self.type = 'dmf'
        self.net = self.init_para(params)
        self.data = self.init_data()
        self.opt = self.init_opt()


    def init_para(self,params):
        # Initial the parameter (Deep linear network)
        hidden_sizes = params
        layers = zip(hidden_sizes, hidden_sizes[1:])
        nn_list = []
        for (f_in,f_out) in layers:
            nn_list.append(nn.Linear(f_in, f_out, bias=False))
        model = nn.Sequential(*nn_list)
        if cuda_if:
            model = model.cuda()
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,mean=1e-3,std=1e-3)
        return model

    def init_data(self):
        # Initial data
        def get_e2e(model):
            #获取预测矩阵
            weight = None
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(weight)
            return weight
        return get_e2e(self.net)

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters())
        return optimizer
    

    
    def update(self):
        self.opt.step()
        self.data = self.init_data()

class dmf_rand(object):
    # Deep Matrix Factorization with random input
    def __init__(self,params):
        self.type = 'dmf_rand'
        self.net = self.init_para(params)
        self.input = t.eye(params[0],params[1])
        self.input = self.input.cuda()
        self.data = self.init_data()
        self.opt = self.init_opt()
        


    def init_para(self,params):
        # Initial the parameter (Deep linear network)
        hidden_sizes = params
        layers = zip(hidden_sizes, hidden_sizes[1:])
        nn_list = []
        for (f_in,f_out) in layers:
            nn_list.append(nn.Linear(f_in, f_out, bias=False))
        model = nn.Sequential(*nn_list)
        if cuda_if:
            model = model.cuda()
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,mean=1e-3,std=1e-3)
        return model

    def init_data(self):
        # Initial data
        def get_e2e(model,input_data):
            #获取预测矩阵
            weight = input_data
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(weight)
            return weight
        return get_e2e(self.net,self.input+t.randn(self.input.shape).cuda()*1e-2)
    
    def show_img(self):
        def get_e2e(model,input_data):
            #获取预测矩阵
            weight = input_data
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(weight)
            return weight
        return get_e2e(self.net,self.input)
    
    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters())
        return optimizer

    def update(self):
        self.opt.step()
        self.data = self.init_data()

class hadm(object):
    # Hadmard Product
    def __init__(self,params,def_type=0,hadm_lr=1e-3):
        self.type = 'hadm'
        self.def_type = def_type
        self.net = self.init_para((params[0],params[-1]))
        self.data = self.init_data()
        self.opt = self.init_opt(hadm_lr=hadm_lr)

    def init_para(self,params):
        # Initial the parameter (Deep linear network)
        g = t.randn(params)*1e-4
        h = t.randn(params)*1e-4
        if cuda_if:
            g = g.cuda()
            h = h.cuda()
        g = Variable(g,requires_grad=True)
        h = Variable(h,requires_grad=True)
        return [g,h]

    def init_data(self):
        # Initial data
        if self.def_type == 0:
            return self.net[0]*self.net[1]
        else:
            return self.net[0]*self.net[0]-self.net[1]*self.net[1]

    def init_opt(self,hadm_lr=1e-3):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net,hadm_lr)
        return optimizer

    def update(self):
        self.opt.step()
        self.data = self.init_data()

class dip(object):
    # unet like neural network, which have DIP
    def __init__(self,params,img,lr=1e-3):
        self.type = 'dip'
        self.net = self.init_para(params)
        self.img = img
        self.input = t.rand(img.shape)*1e-1
        self.input = self.input.cuda()
        self.data = self.init_data(self.input)
        self.opt = self.init_opt(lr)
        


    def init_para(self,params):
        # Initial the parameter (Deep Image Prior)
        input_depth = 32
        pad = 'reflection'
        dtype = torch.cuda.FloatTensor
        net = get_net(input_depth, 'skip', pad,
                      skip_n33d=64, 
                      skip_n33u=64, 
                      skip_n11=4, 
                      num_scales=5,
                      upsample_mode='bilinear',
                      n_channels=1).type(dtype)
        return net

    def init_data(self,input_img):
        # Initial data
        #print(self.img.shape)
        pre_img = self.net(input_img)
        pre_img = t.squeeze(pre_img,dim=0)
        pre_img = t.squeeze(pre_img,dim=0)
        #print(pre_img.shape)
        return pre_img

    def init_opt(self,lr):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters(),lr)
        return optimizer

    def update(self):
        self.opt.step()
        self.data = self.init_data(self.input+t.randn(self.input.shape).cuda()*5e-2)


class nl_dmf(dmf):
    # Nonlinear deep matrix factorization
    def __init__(self,params):
        dmf.__init__(self,params)

    def init_data(self):
        # Initial data
        def get_e2e(model):
            #获取预测矩阵
            weight = None
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(t.sigmoid(weight))
            return t.sigmoid(weight)
        return get_e2e(self.net)

