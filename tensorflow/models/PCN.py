import tensorflow as tf
from net_util import mlp, mlp_conv
from net_util import chamfer, emd
from common import num_params, create_pcn_encoder, create_multigpu_model

def PCN_setup(args):
    args.odir = 'results/%s/PCN_%s' % (args.dataset, args.dist_fun)
    grid_size = {2048: 2, 4096: 2, 8192: 4, 16384: 4}
    args.grid_size = grid_size[args.npts]
    args.num_coarse = int(args.npts/(args.grid_size**2))
    args.npts = (args.grid_size**2) * args.num_coarse
    args.odir += '_npts%d' % (args.npts)
    args.odir += '_grid%d' % (args.grid_size)
    args.odir += '_lr%.4f' % (args.lr)
    args.odir += '_' + args.optim
    args.odir += '_B%d' % (args.batch_size)

def PCN_create_model(args):
    create_multigpu_model(PCN, args)
    args.nparams = num_params()
    print('Total number of parameters: {}'.format(args.nparams))

def PCN_step(args, targets, clouds_data):
    _, loss, dist1, dist2, emd_cost, outputs = args.sess.run([args.train_op, args.loss, args.dist1, args.dist2, args.emd_cost, args.outputs],
                                {args.partial: clouds_data[1].transpose((0, 2, 1)),
                                args.gt: targets, args.phase: args.training})

    return loss, dist1, dist2, emd_cost, outputs

class PCN:
    def __init__(self, args, inputs, gt):
        self.args = args
        self.num_coarse = args.num_coarse
        self.grid_size = args.grid_size
        self.num_fine = (self.grid_size ** 2) * self.num_coarse
        self.features = create_pcn_encoder(inputs, args)
        if not hasattr(args, 'enc_params'):
            args.enc_params = num_params()
        self.coarse, self.fine = self.create_decoder(self.features)
        self.dist1, self.dist2, self.loss, self.emd_cost = self.create_loss(self.coarse, self.fine, gt)
        self.outputs = self.fine

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3], self.args.phase)
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [tf.shape(features)[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            fine = mlp_conv(feat, [512, 512, 3], self.args.phase)
            fine = fine + center
            print(coarse.shape, fine.shape)
        return coarse, fine

    def create_loss(self, coarse, fine, gt):
        _, _, loss_coarse = eval(self.args.dist_fun)(coarse, gt)
        dist1, dist2, loss_fine = eval(self.args.dist_fun)(fine, gt)
        loss = loss_coarse + loss_fine
        emd_cost = emd(fine, gt)

        return dist1, dist2, loss, emd_cost
