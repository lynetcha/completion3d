import tensorflow as tf
from net_util import mlp_conv
from net_util import chamfer, emd
from common import num_params, create_pcn_encoder, create_multigpu_model
import numpy as np

def Folding_setup(args):
    args.odir = 'results/%s/Folding_%s' % (args.dataset, args.dist_fun)
    args.grid_size = int(np.sqrt(args.npts))
    if args.grid_size**2 < args.npts:
        args.grid_size += 1
    args.npts = args.grid_size**2
    args.odir += '_npts%d' % (args.npts)
    args.odir += '_lr%.4f' % (args.lr)
    args.odir += '_' + args.optim
    args.odir += '_B%d' % (args.batch_size)

def Folding_create_model(args):
    create_multigpu_model(Folding, args)
    args.nparams = num_params()
    print('Total number of parameters: {}'.format(args.nparams))
    return None

def Folding_step(args, targets, clouds_data):
    outs = [args.train_op, args.loss, args.dist1, args.dist2, args.emd_cost, args.outputs]
    if not args.training:
        outs = outs[1:]
    loss, dist1, dist2, emd_cost, outputs = args.sess.run(
        outs, {args.partial: clouds_data[1].transpose((0, 2, 1)),
               args.gt: targets, args.phase: args.training})[-5:]

    return loss, dist1, dist2, emd_cost, outputs

class Folding:
    def __init__(self, args, inputs, gt):
        self.args = args
        self.grid_size = args.grid_size
        self.grid_scale = 0.5
        self.num_output_points = args.npts
        self.features = create_pcn_encoder(inputs, args)
        if not hasattr(args, 'enc_params'):
            args.enc_params = num_params()
        fold1, fold2 = self.create_decoder(self.features)
        self.outputs = fold2
        self.dist1, self.dist2, self.loss, self.emd_cost = self.create_loss(self.outputs, gt)

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            grid = tf.meshgrid(x, y)
            grid = tf.reshape(tf.stack(grid, axis=2), [-1, 2])
            grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(features)[0], 1, 1])
            features = tf.tile(tf.expand_dims(features, 1), [1, self.num_output_points, 1])
            with tf.variable_scope('folding_1'):
                fold1 = mlp_conv(tf.concat([features, grid], axis=2), [512, 512, 3], self.args.phase)
            with tf.variable_scope('folding_2'):
                fold2 = mlp_conv(tf.concat([features, fold1], axis=2), [512, 512, 3], self.args.phase)
            print(fold1.shape, fold2.shape)
        return fold1, fold2

    def create_loss(self, outputs, gt):
        dist1, dist2, loss = eval(self.args.dist_fun)(outputs, gt)
        emd_cost =  emd(outputs[:, 0:gt.shape[1],:], gt)

        return dist1, dist2, loss, emd_cost


