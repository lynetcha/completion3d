import tensorflow as tf
from net_util import mlp, mlp_conv
from net_util import chamfer, emd
from common import num_params, create_multigpu_model, encoders
import numpy as np
import math

# Number of children per tree levels for 2048 output points
tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

def get_arch(nlevels, npts):
    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch

print(get_arch(8, 16384))

def TopNet_setup(args):
    args.odir = 'results/%s/TopNet%dL%dF%d_%s' % (args.dataset, args.ENCODER_ID, args.NLEVELS, args.NFEAT, args.dist_fun)
    args.tarch = get_arch(args.NLEVELS, args.npts)
    N = int(np.prod([int(k) for k in args.tarch]))
    assert N == args.npts, "Number of tree outputs is %d, expected %d" % (N, args.npts)
    args.odir += '_npts%d' % (args.npts)
    args.odir += '_lr%.4f' % (args.lr)
    args.odir += '_' + args.optim
    args.odir += '_B%d' % (args.batch_size)

def TopNet_create_model(args):
    create_multigpu_model(TopNet, args)
    args.nparams = num_params()
    print('Total number of parameters: {}'.format(args.nparams))

def TopNet_step(args, targets, clouds_data):
    _, loss, dist1, dist2, emd_cost, outputs = args.sess.run([args.train_op, args.loss, args.dist1, args.dist2, args.emd_cost, args.outputs],
                                {args.partial: clouds_data[1].transpose((0, 2, 1)),
                                 args.gt: targets, args.phase: args.training})

    return loss, dist1, dist2, emd_cost, outputs


class TopNet:
    def __init__(self, args, inputs, gt):
        self.args = args
        self.features = encoders[args.ENCODER_ID](inputs, args)
        if not hasattr(args, 'enc_params'):
            args.enc_params = num_params()
        self.outputs = self.create_decoder(self.features)
        self.dist1, self.dist2, self.loss, self.emd_cost = self.create_loss(self.outputs, gt)

    def create_level(self, level, input_channels, output_channels, inputs, bn):
        with tf.variable_scope('level_%d' % (level), reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [input_channels, int(input_channels/2),
                                         int(input_channels/4), int(input_channels/8),
                                         output_channels*int(self.args.tarch[level])],
                                        self.args.phase, bn)
            features = tf.reshape(features, [tf.shape(features)[0], -1, output_channels])
        return features

    def create_decoder(self, code):
        Nin = self.args.NFEAT + self.args.code_nfts
        Nout = self.args.NFEAT
        bn = True
        N0 = int(self.args.tarch[0])
        nlevels = len(self.args.tarch)
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            level0 = mlp(code, [256, 64, self.args.NFEAT * N0], self.args.phase, bn=True)
            level0 = tf.tanh(level0, name='tanh_0')
            level0 = tf.reshape(level0, [-1, N0, self.args.NFEAT])
            outs = [level0, ]
            for i in range(1, nlevels):
                if i == nlevels - 1:
                    Nout = 3
                    bn = False
                inp = outs[-1]
                y = tf.expand_dims(code, 1)
                y = tf.tile(y, [1, tf.shape(inp)[1], 1])
                y = tf.concat([inp, y], 2)
                outs.append(tf.tanh(self.create_level(i, Nin, Nout, y, bn), name='tanh_%d' % (i)))
        return outs[-1]

    def create_loss(self, outputs, gt):
        emd_cost = emd(outputs, gt)
        dist1, dist2, loss = eval(self.args.dist_fun)(outputs, gt)
        return dist1, dist2, loss, emd_cost
