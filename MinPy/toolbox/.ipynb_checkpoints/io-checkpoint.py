import pickle as pkl

def save(model,path,lap_if = True,pic_if=True):
    save_dict = {}
    #save_dict['net'] = model.net
    save_dict['loss'] = model.loss_dict
    if lap_if:
        save_dict['lap_dict'] = model.lap_dict
    if pic_if:
        save_dict['pic_list'] = model.pic_list
    with open(path,'wb') as f:
        pkl.dump(save_dict,f)
    print('Success save '+path)
    
def load(path,verbose=True):
    with open(path,'rb') as f:
        data = pkl.load(f)
    if verbose:
        print('Success load '+path)
    return data

