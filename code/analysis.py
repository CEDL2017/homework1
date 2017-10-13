
import os
import argparse

import pdb
# pdb.set_trace = lambda: None
import itertools
import scipy.io as sio
import zipfile
import numpy as np
import shutil

# plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
'''
python analysis.py -n r_37_2_lr1e-1_e200,r_37_2_lr1e-1_best,r_37_2_lr1e-2_e200,r_37_2_lr1e-2_best --prc 
'''

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 analysis')
parser.add_argument('-e', '--eval_dir', default="", type=str, help='eval_dir')
parser.add_argument('-n', '--net', default="", type=str, help='net name')
parser.add_argument('--dataset_split_name', default="", type=str, help='split name')
parser.add_argument('--cls', '-c', default=2, type=int, help='cls 0:fa, 1:ges, 2:obj')
parser.add_argument('--gr', action='store_true', help='plot group')
args = parser.parse_args()

cls = [['free',
        'active',],
    ['free',
    'press',
    'large-diameter',
    'lateral-tripod',
    'parallel-extension',
    'thumb-2-finger',
    'thumb-4-finger',
    'thumb-index-finger',
    'precision-disk',
    'lateral-pinch',
    'tripod',
    'medium-wrap',
    'light-tool'],
    ['free',
    'computer',
    'cellphone',
    'coin',
    'ruler',
    'thermos-bottle',
    'whiteboard-pen',
    'whiteboard-eraser',
    'pen',
    'cup',
    'remote-control-TV',
    'remote-control-AC',
    'switch',
    'windows',
    'fridge',
    'cupboard',
    'water-tap',
    'toy',
    'kettle',
    'bottle',
    'cookie',
    'book',
    'magnet',
    'lamp-switch']]

def pr_curve(pr_point, pr_auc):
    # load pred
    print(args.net+"_: (recall, prec)")
    
    # plot
    lines = []
    labels = []

    l, = plt.plot(pr_point[:,0], pr_point[:,1], lw=2)
    lines.append(l)
    labels.append('Precision-recall (area = {0:0.2f})'
                  ''.format(pr_auc))
    fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.75)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve to multi-class')
    # save
    sdir = os.path.join('plot',args.net)
    if not os.path.isdir(sdir):
        os.makedirs(sdir)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir,"plot_prc.png"), bbox_inches='tight', dpi=fig.dpi)
    plt.close(fig)
 
 
def ccmatrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # load pred
    print(args.net+"_: (ccmatrix)")
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    fig = plt.gcf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    '''
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    '''
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    sdir = os.path.join('plot',args.net)
    if not os.path.isdir(sdir):
        os.makedirs(sdir)
    plt.savefig(os.path.join(sdir,"plot_cm.png"), bbox_inches='tight', dpi=fig.dpi)
    plt.close(fig)

def main():
    plt.ioff() 
    dict_data = np.load(os.path.join(args.eval_dir, args.dataset_split_name+'_rst%d.npy'%args.cls))
    pr_point = dict_data.all()['pr_point']
    pr_auc = dict_data.all()['pr_auc']
    c_matrix = dict_data.all()['confusion_matrix']
    classes = cls[args.cls]
    pr_curve(pr_point, pr_auc)
    ccmatrix(c_matrix, classes, normalize=True, cmap=plt.cm.YlGnBu)
    
if __name__ == '__main__':
    main()