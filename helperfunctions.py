import os
import shutil

import numpy as np



def save_img(label,masked=None,img=None,mask=None,savename="",save_dir=os.getcwd()):

    filename = f"{label}_"
    if masked is not None:
        path = os.path.join(save_dir, filename + "masked_" + savename + ".npy")
        np.save(path, masked)

    elif img is not None:
        path = os.path.join(save_dir, filename + "img_" + savename + ".npy")
        np.save(path, img)

    elif mask is not None:
        path = os.path.join(save_dir, filename + "mask_" + savename + ".npy")
        np.save(path, mask)

def mkdir(root_dir, dirname="patches"):

    dir = os.path.join(root_dir, dirname)

    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir
def mkresultsdir(training_dir):
    return mkdir(training_dir, dirname="results")

def mkepochdir(training_dir, epoch):

    save_dir = mkresultsdir(training_dir)

    epochdir = os.path.join(save_dir, f"epoch_{epoch}")

    if not os.path.exists(epochdir):
        os.mkdir(epochdir)

    return epochdir

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
