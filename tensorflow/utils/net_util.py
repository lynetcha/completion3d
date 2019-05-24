import tensorflow as tf
from pc_distance import tf_nndistance, tf_approxmatch


def mlp(features, layer_dims, phase, bn=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            activation_fn=None,
            normalizer_fn=None,
            scope='fc_%d' % i)
        if bn:
            with tf.variable_scope('fc_bn_%d' % (i), reuse=tf.AUTO_REUSE):
                features = tf.layers.batch_normalization(features, training=phase)
        features = tf.nn.relu(features, 'fc_relu_%d' % i)

    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, phase, bn=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            activation_fn=None,
            normalizer_fn=None,
            scope='conv_%d' % i)
        if bn:
            with tf.variable_scope('conv_bn_%d' % (i), reuse=tf.AUTO_REUSE):
                inputs = tf.layers.batch_normalization(inputs, training=phase)
        inputs = tf.nn.relu(inputs, 'conv_relu_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs

def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    mdist1 = tf.reduce_mean(dist1)
    mdist2 = tf.reduce_mean(dist2)
    return dist1, dist2, (mdist1 + mdist2)

def emd(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return cost / num_points
