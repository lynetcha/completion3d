import numpy as np
import tensorflow as tf
from net_util import mlp_conv, mlp

def create_pointnet_encoder(inputs, args):
    with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
        features = mlp_conv(inputs, [64, 128, args.code_nfts], args.phase, bn=True)
        code = tf.reduce_max(features, axis=1, name='maxpool_0')
    with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
        code = mlp(code, [args.code_nfts,], args.phase)
        with tf.variable_scope('bn', reuse=tf.AUTO_REUSE):
            code = tf.layers.batch_normalization(code, training=args.phase)
        code = tf.nn.relu(code, 'relu')
    return code

def create_pcn_encoder(inputs, args):
    with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
        features = mlp_conv(inputs, [128, 256], args.phase)
        features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
        features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
    with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
        features = mlp_conv(features, [512, args.code_nfts], args.phase)
        features = tf.reduce_max(features, axis=1, name='maxpool_1')
    return features

def num_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

def create_multigpu_model(ModelClass, args):
    tower_grads = []
    outputs = []
    losses = []
    dist1, dist2 = [], []
    emd_cost = []
    assert args.batch_size % args.num_gpus == 0, \
        "%d vs % d Batch size must be multiple of number of GPUs" % (
            args.batch_size, args.num_gpus)
    B = int(args.batch_size/args.num_gpus)
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i in range(args.num_gpus):
            with tf.device('/gpu:%d' % i):
                partial_i = tf.slice(args.partial, [i*B, 0, 0], [B, -1, -1])
                gt_i = tf.slice(args.gt, [i*B, 0, 0], [B, -1, -1])
                model = ModelClass(args, partial_i, gt_i)
                loss_i = model.loss
                dist1.append(model.dist1)
                dist2.append(model.dist2)
                outputs_i = model.outputs
                outer_scope.reuse_variables()
                grads_i = args.optimizer.compute_gradients(loss_i)
                tower_grads.append(grads_i)
                losses.append(loss_i)
                outputs.append(outputs_i)
                emd_cost.append(model.emd_cost)
    grads = average_gradients(tower_grads)
    global_step = tf.train.get_or_create_global_step()
    args.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(args.extra_update_ops):
        args.train_op = args.optimizer.apply_gradients(grads, global_step)
    args.loss = tf.reduce_mean(losses)
    args.outputs = tf.concat(outputs, 0)
    args.dist1 = tf.concat(dist1, 0)
    args.dist2 = tf.concat(dist2, 0)
    args.emd_cost = tf.concat(emd_cost, 0)

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


encoders = {0: create_pointnet_encoder, 1: create_pcn_encoder}
