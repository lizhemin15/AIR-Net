import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import misc
def gray_im(img,path='./test.jpg',save_if=False):
    show_pic = np.clip(img,0,1)
    plt.imshow(show_pic,'gray',vmin=0,vmax=1)
    plt.grid(0)
    plt.axis('off')
    plt.show()
    if save_if:
        misc.imsave(path, show_pic)


def lines(line_dict,xlabel_name='epoch',ylabel_name='MSE',ylog_if=False,save_if=False,path='./lines.jpg',black_if=False):
    if black_if:
        sns.set()
    else:
        sns.set_style("whitegrid")  
    for name in line_dict.keys():
        if name != 'x_plot':
            plt.plot(line_dict['x_plot'],line_dict[name],label=name)
    plt.legend()
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    if ylog_if:
        plt.yscale('log')
    if save_if:
        plt.savefig(path)
    plt.show()


