
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from ..config import settings



import torch as t
import numpy as np
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import cv2
import random


def get_data(width=100,height=100,pic_name=None):
    # 合成或从路径读取图像
    # 返回值为tensor类型的数据，大小为length*width
    if pic_name == None:
        x = np.squeeze(np.linspace(-1, 1, width))
        y = np.squeeze(np.linspace(-1, 1, height))
        x1,y1 = np.meshgrid(x,y)
        z = np.sin(25*np.pi*np.sin(np.pi/3*np.sqrt(x1**2+y1**2)))
        z = z.astype('float32')/z.max()
    else:
        img = cv2.imread(pic_name)
        img = cv2.resize(img, (height,width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        z = img.astype('float32')/255

    return t.tensor(z)

    
class data_transform(object):
    # 几种数据变换的类型，从tensor进行变换,根据需求返回tensor或Dataloader
    # shuffle: 定义的最大函数类
    # drop: 丢掉一部分像素点
    # noise: 加噪声
    def __init__(self,z=None,return_type='tensor'):
        self.z = z
        self.height = self.z.shape[0]
        self.width = self.z.shape[1]
        self.dataloader = self.transform_z(return_type)
        self.x,self.y = self.test_x()
        
    def test_x(self):
        x = t.squeeze(t.linspace(-1, 1, self.width), dim=0)
        y = t.squeeze(t.linspace(-1, 1, self.height), dim=0)
        x1,y1 = t.meshgrid(x,y)
        x1 = t.unsqueeze(x1,dim=2)
        y1 = t.unsqueeze(y1,dim=2)
        x = t.cat((x1,y1),dim=2)
        x = x.reshape(-1,2)
        y = self.z.reshape(-1,1)
        x, y = Variable(x), Variable(y)
        return x,y
    
    def transform_z(self,return_type=None,batch_size=256):
        if return_type == 'tensor':
            return self.z
        elif return_type == 'dataloader':
            x = t.squeeze(t.linspace(-1, 1, self.width), dim=0)
            y = t.squeeze(t.linspace(-1, 1, self.height), dim=0)
            x1,y1 = t.meshgrid(x,y)
            x1 = t.unsqueeze(x1,dim=2)
            y1 = t.unsqueeze(y1,dim=2)
            x = t.cat((x1,y1),dim=2)
            x = x.reshape(-1,2)
            y = self.z.reshape(-1,1)
            x, y = Variable(x), Variable(y)
            dataset=TensorDataset(x.clone().detach(),y.clone().detach())
            dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
            return dataloader
        else:
            raise('Wrong return_type, your type is ',return_type)
    
    def get_shuffle_list(self,mode='I'):
        # 返回一个长度为 length*width 的list
        # 根据这个list对输入图像进行变换
        shuffle_matrix = t.tensor(np.arange(self.height*self.width).reshape((self.height,self.width)))
        shuffle_matrix = shuffle_matrix.float()
        data_shuffler = data_shuffle(M=shuffle_matrix,mode=mode)
        shuffle_matrix = data_shuffler.shuffle_M
        shuffle_vec = shuffle_matrix.reshape((-1,1))
        shuffle_vec = shuffle_vec.int()
        return list(shuffle_vec)
    
    def shuffle(self,M=None,shuffle_list=None,mode='from',return_type='tensor',batch_size=256):
        # 输入M为待shuffle的tensor矩阵，list_shuffle为之前得到的shuffle_list
        # mode决定了是进行shuffle还是还原，M会随之变化
        list_vec = list(M.reshape((-1,1)))
        list_new = list_vec.copy()
        if mode == 'from':
            for i,item in enumerate(shuffle_list):
                list_new[item] = list_vec[i]
        else:
            for i,item in enumerate(shuffle_list):
                list_new[i] = list_vec[item]
        if return_type == 'tensor':
            return t.tensor(list_new).reshape((self.height,self.width))
        elif return_type == 'numpy':
            return t.tensor(list_new).reshape((self.height,self.width)).numpy()
        elif return_type == 'dataloader':
            x = t.squeeze(t.linspace(-1, 1, self.width), dim=0)
            y = t.squeeze(t.linspace(-1, 1, self.height), dim=0)
            x1,y1 = t.meshgrid(x,y)
            x1 = t.unsqueeze(x1,dim=2)
            y1 = t.unsqueeze(y1,dim=2)
            x = t.cat((x1,y1),dim=2)
            x = x.reshape(-1,2)
            y = t.tensor(list_new).reshape(-1,1)
            x, y = Variable(x), Variable(y)
            dataset = TensorDataset(x.clone().detach(),y.clone().detach())
            dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
            return dataloader
        else:
            raise('Wrong type')
    
    def get_drop_mask(self,rate=0):
        # 返回一个形状为 (length,width) 的matrix
        drop_matrix = t.tensor(np.random.random((self.height,self.width))>rate).int()
        return drop_matrix
    
    def drop(self,M=None,drop_matrix=None,return_type='tensor',batch_size=256):
        # 输入一个 drop_matrix 和待drop矩阵M
        if return_type == 'tensor':
            return drop_matrix*M
        elif return_type == 'numpy':
            return (drop_matrix*M).numpy()
        elif return_type == 'dataloader':
            x = t.squeeze(t.linspace(-1, 1, self.width), dim=0)
            y = t.squeeze(t.linspace(-1, 1, self.height), dim=0)
            x1,y1 = t.meshgrid(x,y)
            x1 = t.unsqueeze(x1,dim=2)
            y1 = t.unsqueeze(y1,dim=2)
            x = t.cat((x1,y1),dim=2)
            x = x.reshape(-1,2)
            y = M.reshape(-1,1)
            index = t.where(drop_matrix.reshape(-1,1) == 1)[0]
            x = x[index]
            y = y[index]
            x, y = Variable(x), Variable(y)
            dataset=TensorDataset(x.clone().detach(),y.clone().detach())
            dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
            return dataloader
        else:
            raise('Wrong type')

    def add_noise(self,M=None,rate=0,return_type='tensor',batch_size=256):
        # 往数据M中加噪声
        if return_type == 'tensor':
            return t.clamp(M+t.tensor(np.random.random((self.height,self.width))-0.5)*rate,0,1)
        elif return_type == 'numpy':
            return t.clamp(M+t.tensor(np.random.random((self.height,self.width))-0.5)*rate,0,1).numpy()
        elif return_type == 'dataloader':
            x = t.squeeze(t.linspace(-1, 1, self.width), dim=0)
            y = t.squeeze(t.linspace(-1, 1, self.height), dim=0)
            x1,y1 = t.meshgrid(x,y)
            x1 = t.unsqueeze(x1,dim=2)
            y1 = t.unsqueeze(y1,dim=2)
            x = t.cat((x1,y1),dim=2)
            x = x.reshape(-1,2)
            y = t.clamp(M+t.tensor(np.random.random((self.height,self.width))-0.5)*rate,0,1).reshape(-1,1)
            x, y = Variable(x), Variable(y)
            dataset=TensorDataset(x.clone().detach(),y.clone().detach())
            dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
            return dataloader
        else:
            raise('Wring type')


class data_shuffle(object):
    # 将输入图像像素点进行打乱，分四种变换类进行
    # 1.\mathcal{A} 随机重排所有像素点
    # 2.\mathcal{E} 初等置换矩阵作用于矩阵
    # 3.\mathcal{C} 上下左右循环平移
    # 4.\mathcal{I} 原图
    def __init__(self,M=None,mode='I'):
        self.M = M
        self.m = M.shape[0]
        self.n = M.shape[1]
        self.mode = mode
        exec('self.shuffle_M=self.'+self.mode+'()')
    
    def J(self,m):
        My_J = t.eye(m)
        My_J[:-1] = My_J[1:].clone()
        My_J[-1,-1] = 0
        My_J[-1,0] = 1
        return My_J
    
    def I(self):
        self.shuffle_M = self.M
        return self.M
    
    def C(self):
        # 循环式置换矩阵
        down_num = np.random.randint(self.m)
        right_num = np.random.randint(self.n)
        left_J = t.eye(self.m)
        right_J = t.eye(self.n)
        for i in range(down_num):
            left_J = left_J.matmul(self.J(self.m))
        for i in range(right_num):
            right_J = right_J.matmul(self.J(self.n))
        shuffle_M = left_J.matmul(self.M)
        shuffle_M = shuffle_M.matmul(right_J.T)
        self.left = left_J
        self.right = right_J
        self.shuffle_M = shuffle_M
        return shuffle_M
    
    def E(self):
        # 行列随机交换
        E1_list = list(t.eye(self.m))
        random.shuffle(E1_list)
        E1 = t.eye(self.m)
        for item,i in enumerate(E1_list):
            E1[item] = i
        E2_list = list(t.eye(self.n))
        random.shuffle(E2_list)
        E2 = t.eye(self.n)
        for item,i in enumerate(E2_list):
            E2[item] = i
        shuffle_M = E1.matmul(self.M)
        shuffle_M = shuffle_M.matmul(E2.T)
        self.left = E1
        self.right = E2
        self.shuffle_M = shuffle_M
        return shuffle_M
    
    def A(self):
        # 随机交换任意两个元素
        list_vec_M = list(self.M.reshape((-1,1)))
        list_shuffle_matrix = list(np.arange(self.m*self.n))
        random.shuffle(list_shuffle_matrix)
        list_vec_M = self.A_shuffle(list_vec=list_vec_M,list_shuffle=list_shuffle_matrix,mode='from')
        vec_M = t.tensor(list_vec_M)
        self.list_shuffle_matrix = list_shuffle_matrix
        shuffle_M = vec_M.reshape((self.m,self.n))
        self.shuffle_M = shuffle_M
        return shuffle_M
        
    
    def A_shuffle(self,list_vec=None,list_shuffle=None,mode='from'):
        # 输入list
        list_new = list_vec.copy()
        if mode == 'from':
            for i,item in enumerate(list_shuffle):
                list_new[item] = list_vec[i]
        else:
            for i,item in enumerate(list_shuffle):
                list_new[i] = list_vec[item]
        return list_new
    
    
    def back(self):
        if self.mode == 'I':
            return self.shuffle_M
        elif self.mode == 'C' or self.mode == 'E':
            M = self.left.T.matmul(self.shuffle_M)
            return M.matmul(self.right)
        elif self.mode == 'A':
            list_vec_M = list(self.shuffle_M.reshape((-1,1)))
            list_vec_M = self.A_shuffle(list_vec=list_vec_M,list_shuffle=self.list_shuffle_matrix,mode='to')
            vec_M = t.tensor(list_vec_M)
            self.list_vec_M = list_vec_M
            return vec_M.reshape((self.m,self.n))
        else:
            raise('wrong mode')


def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    import h5py
    import scipy.sparse as sp
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out


def load_syn(path):
    path_dataset = path

    W_rows = load_matlab_file(path_dataset, 'Wrow').todense()  # Row Graph
    W_cols = load_matlab_file(path_dataset, 'Wcol').todense()  # Column Graph
    M = load_matlab_file(path_dataset, 'M')
    return W_rows,W_cols,M

def load_douban(path):
    path_dataset = path

    W_users = load_matlab_file(path_dataset, 'W_users')
    O_train = load_matlab_file(path_dataset, 'Otraining')
    O_test = load_matlab_file(path_dataset, 'Otest')
    M = load_matlab_file(path_dataset, 'M')
    return O_train,O_test,W_users,M

def load_drug(path,name='e'):
    import scipy.io as scio
    data = scio.loadmat(path)
    return data
    