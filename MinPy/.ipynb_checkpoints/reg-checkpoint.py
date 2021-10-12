import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t
import numpy as np

cuda_if = settings.cuda_if

class hc_reg(object):
    #使用torch写的正则化项
    #handcraft, hd
    def __init__(self,name='lap',kernel=None,p=2):
        self.__name = name
        self.__kernel = kernel
        self.__p = p
        self.type = 'hc_reg'

    def loss(self,M):
        self.__M = M
        if self.__name == 'tv1':
            return self.tv(p=1)
        elif self.__name == 'tv2':
            return self.tv(p=2)
        elif self.__name == 'lap':
            return self.lap()
        elif self.__name == 'kernel':
            return self.reg_kernel(kernel=self.__kernel,p=self.__p)
        elif self.__name == 'de_row':
            return self.de('row')
        elif self.__name == 'de_col':
            return self.de('col')
        else:
            raise('Please check out your regularization term')
    
    def tv(self,p):
        center = self.__M[1:self.__M.shape[0]-1,1:self.__M.shape[1]-1]
        up = self.__M[1:self.__M.shape[0]-1,0:self.__M.shape[1]-2]
        down = self.__M[1:self.__M.shape[0]-1,2:self.__M.shape[1]]
        left = self.__M[0:self.__M.shape[0]-2,1:self.__M.shape[1]-1]
        right = self.__M[2:self.__M.shape[0],1:self.__M.shape[1]-1]
        Var = 4*center-up-down-left-right
        return t.norm(Var,p=p)/self.__M.shape[0]

            
    def lap(self):
        center = self.__M[1:self.__M.shape[0]-1,1:self.__M.shape[1]-1]
        up = self.__M[1:self.__M.shape[0]-1,0:self.__M.shape[1]-2]
        down = self.__M[1:self.__M.shape[0]-1,2:self.__M.shape[1]]
        left = self.__M[0:self.__M.shape[0]-2,1:self.__M.shape[1]-1]
        right = self.__M[2:self.__M.shape[0],1:self.__M.shape[1]-1]
        Var = 4*center-up-down-left-right
        return t.norm(Var,p=2)/self.__M.shape[0]
    
    def reg_kernel(self,kernel,p=2):
        center = self.__M[1:self.__M.shape[0]-1,1:self.__M.shape[1]-1]
        up = self.__M[1:self.__M.shape[0]-1,0:self.__M.shape[1]-2]
        down = self.__M[1:self.__M.shape[0]-1,2:self.__M.shape[1]]
        left = self.__M[0:self.__M.shape[0]-2,1:self.__M.shape[1]-1]
        right = self.__M[2:self.__M.shape[0],1:self.__M.shape[1]-1]
        lu = self.__M[0:self.__M.shape[0]-2,0:self.__M.shape[1]-2]
        ru = self.__M[2:self.__M.shape[0],0:self.__M.shape[1]-2]
        ld = self.__M[0:self.__M.shape[0]-2,1:self.__M.shape[1]-1]
        rd = self.__M[2:self.__M.shape[0],1:self.__M.shape[1]-1]
        Var = kernel[0][0]*lu+kernel[0][1]*up+kernel[0][2]*ru\
            +kernel[1][0]*left+kernel[1][1]*center+kernel[1][2]*right\
            +kernel[2][0]*ld+kernel[2][1]*down+kernel[2][2]*rd
        return t.norm(Var,p=p)/self.__M.shape[0]*8

    def de(self,mode='row'):
        if mode == 'col':
            M = self.__M.T
        else:
            M = self.__M
        Ones = t.ones(M.shape[1],1)
        Eyes = t.eye(M.shape[0])
        if cuda_if:
            Ones = Ones.cuda()
            Eyes = Eyes.cuda()
        V_M = t.sqrt(t.mm(M**2,Ones))
        cov = t.mm(M,M.T)/t.mm(V_M,V_M.T)
        lap = -cov+2*Eyes
        self.LAP = lap
        return t.trace(t.mm(M.T,t.mm(lap,M)))

        

class auto_reg(object):
    def __init__(self,size,mode='row'):
        self.type = 'auto_reg_'+mode
        if mode == 'row':
            self.net = self.init_net(size,mode)
        else:
            self.net = self.init_net(size,mode)
        if cuda_if:
            self.net = self.net.cuda()
        self.opt = self.init_opt()

    def init_net(self,n,mode='row'):
        class net(nn.Module):
            def __init__(self,n,mode='row'):
                super(net,self).__init__()
                self.n = n
                self.A_0 = nn.Linear(n,n,bias=False)
                self.softmax = nn.Softmax(1)
                self.mode = mode

            def forward(self,W):
                Ones = t.ones(self.n,1)
                I_n = t.from_numpy(np.eye(self.n)).to(t.float32)
                if cuda_if:
                    Ones = Ones.cuda()
                    I_n = I_n.cuda()
                A_0 = self.A_0.weight # A_0 \in \mathbb{R}^{n \times n}
                A_1 = self.softmax(A_0) # A_1 中的元素的取值 \in (0,1) 和为1
                A_2 = (A_1+A_1.T)/2 # A_2 一定是对称的
                A_3 = A_2 * (t.mm(Ones,Ones.T)-I_n) # A_3 将中间的元素都归零，作为邻接矩阵
                A_4 = -A_3+t.mm(A_3,t.mm(Ones,Ones.T))*I_n # A_4 将邻接矩阵转化为拉普拉斯矩阵
                self.lap = A_4

                if self.mode == 'row':
                    return t.trace(t.mm(W.T,t.mm(A_4,W)))#+l1 #行关系
                elif self.mode == 'col':
                    return t.trace(t.mm(W,t.mm(A_4,W.T)))#+l1 #列关系
                elif self.mode == 'all':
                    return t.trace(t.mm(W.T,t.mm(self.A_0.weight,W)))#+l1 #所有L
        return net(n,mode)

    def update(self,W):
        self.opt.step()
        self.data = self.init_data(W)


    def init_data(self,W):
        return self.net(W)

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters())
        return optimizer

class cnn_reg(object):
    def __init__(self):
        self.type = 'cnn_reg'
        self.net = self.init_net()
        if cuda_if:
            self.net = self.net.cuda()
        self.opt = self.init_opt()


    def init_net(self):
        class kernel(nn.Module):
            def __init__(self):
                super(kernel,self).__init__()
                self.K_0 = nn.Linear(3,3,bias=False)
                
                
            def forward(self):
                
                Ones = t.ones(3,1)
                I_n = t.from_numpy(np.eye(3)).to(t.float32)
                if cuda_if:
                    Ones = Ones.cuda()
                    I_n = I_n.cuda()
                K_1 = t.sigmoid(self.K_0.weight) # 使得K_1 中元素大于等于零小于等于一
                inner_12 = t.tensor([[1,1,1],[1,0,1],[1,1,1]]).to(t.float32)
                if cuda_if:
                    inner_12 = inner_12.cuda()
                K_2 = -inner_12 * K_1 # 挖空中间的元素，并取负
                inner_23 = t.tensor([[0,0,0],[0,1,0],[0,0,0]]).to(t.float32)
                if cuda_if:
                    inner_23 = inner_23.cuda()
                K_3 = K_2 - inner_23 * (t.mm(Ones.T,t.mm(K_2,Ones))) # 向中间填充进周围元素和相反数
                K_4 = K_3/t.norm(K_3,p='fro') # 将K_3归一化，防止收敛到0矩阵
                self.K_4 = K_4
                return K_4
            
        class net(nn.Module):
            def __init__(self):
                super(net,self).__init__()
                self.Kernel = kernel()
                
            def forward(self,M):
                return t.norm(nn.functional.conv2d(M.unsqueeze(dim=0).unsqueeze(dim=1),self.Kernel().unsqueeze(dim=0).unsqueeze(dim=1)),p=1)

        return net()


    def init_data(self,W):
        return self.net(W)

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.SGD(self.net.parameters(),lr=1e-1)
        return optimizer

    def update(self,W):
        self.opt.step()
        self.data = self.init_data(W)