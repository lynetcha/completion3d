import h5py
import numpy as np
import pandas as pd
import transforms3d
import random
import math

def pad_cloudN(P, Nin):
    """ Pad or subsample 3D Point cloud to Nin number of points """
    N = P.shape[0]
    P = P[:].astype(np.float32)

    rs = np.random.random.__self__
    choice = np.arange(N)
    if N > Nin: # need to subsample
        ii = rs.choice(N, Nin)
        choice = ii
    elif N < Nin: # need to pad by duplication
        ii = rs.choice(N, Nin - N)
        choice = np.concatenate([range(N),ii])
    P = P[choice, :]

    return P

def augment_cloud(Ps, args):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if args.pc_augm_scale > 1:
        s = random.uniform(1/args.pc_augm_scale, args.pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if args.pc_augm_rot:
        angle = random.uniform(0, 2*math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) # y=upright assumption
    if args.pc_augm_mirror_prob > 0: # mirroring x&z, not y
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)
    result = []
    for P in Ps:
        P[:,:3] = np.dot(P[:,:3], M.T)

        if args.pc_augm_jitter:
            sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
            P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
        result.append(P)
    return result

def load_h5(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    f = h5py.File(path, 'r')
    cloud_data = np.array(f['data'])
    f.close()

    return cloud_data.astype(np.float64)

def load_csv(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    return pd.read_csv(path, delim_whitespace=True, header=None).values

