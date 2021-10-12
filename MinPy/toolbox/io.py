import pickle as pkl

def save(model,path):
    save_dict = {}
    save_dict['net'] = model.net
    save_dict['loss'] = model.loss_dict
    with open(path,'wb') as f:
        pkl.dump(model.net,f)
    print('Success save '+path)
    
def load(path):
    with open(path,'rb') as f:
        data = pkl.load(f)
    print('Success load '+path)
    return data

