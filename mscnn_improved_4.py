import re
import tensorflow as tf
import numpy as np
from densityDetection import mscnn_train
import warnings

# variables
MP_NAME = 'mp'
train_log = 'train_log'
model = 'model'
output = 'output'
data_train_gt = 'Data_modified_cropped/Data_gt/1/'
data_train_im = 'Data_modified_cropped/Data_im/1/'
data_train_index = 'Data_modified_cropped/dir_name.txt'
# data_train_gt = 'Data_original/Data_gt/train_gt/'
# data_train_im = 'Data_original/Data_im/train_im/'
# data_train_index = 'Data_original/dir_name.txt'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1, """batch size""")
tf.app.flags.DEFINE_string('train_log', train_log, """train log""")
tf.app.flags.DEFINE_string('model_dir', model, """saving model""")
tf.app.flags.DEFINE_string('output_dir', output, """output""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """is it logged""")
tf.app.flags.DEFINE_string('data_train_gt', data_train_gt, """training labels""")
tf.app.flags.DEFINE_string('data_train_im', data_train_im, """training images""")
tf.app.flags.DEFINE_string('data_train_index', data_train_index, """training image indexes""")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def _activation_summary(x):
    """
    summary function
    :param x: variable waiting to be saved
    :return: None
    """
    tensor_name = re.sub('%s_[0-9]*/' % MP_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """
    Create variables
    :param name: name_scope
    :param shape: tensor dimensions
    :param initializer: initial value
    :return: tensor variable
    """
    with tf.device('/device:GPU:0'):
        var = tf.get_variable(name, shape, initializer=initializer)

    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    create a variable with weight decay
    :param name: name_scope
    :param shape: tensor dimentions
    :param stddev: standard deviation used to initialize
    :param wd: weight decay if == none: it has no decay index
    :return: tensor variable
    """
    var = _variable_on_cpu(name, shape, tf.random_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


class BatchNorm(object):
    """
    Batch normalization
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """
        initializing function
        :param epsilon: precition
        :param momentum: momentum factor
        :param name: name_scope
        """
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x):
        """
        batch norm
        :param x: input variable
        :return: batch_normed variable
        """
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None,
                                            epsilon=self.epsilon, scale=True, scope=self.name)


def multi_scale_block(in_con, in_dim, out_dim, is_bn=False):
    """
    multi-scale block
    :param in_con: input tensor [batch_size, filter_w, filter_h, in_dim]
    :param in_dim: input dimensions
    :param out_dim: output dimensions
    :param is_bn: is add Batch Normal
    :return: output tensor [4 * batch_size, filter_w, filter_h, in_dim]
    """
    with tf.variable_scope('con_9') as scope:
        # reduce channel size
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, in_dim, out_dim], stddev=0.01, wd=0.0005)
        con_9 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)

        kernel = _variable_with_weight_decay('weights_1', shape=[1, 9, out_dim, out_dim], stddev=0.01, wd=0.0005)
        con_9 = tf.nn.conv2d(con_9, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)

        kernel = _variable_with_weight_decay('weights_2', shape=[9, 1, out_dim, out_dim], stddev=0.01, wd=0.0005)
        con_9 = tf.nn.conv2d(con_9, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_9)

    with tf.variable_scope('con_7') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, in_dim, out_dim], stddev=0.01, wd=0.0005)
        con_7 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)

        kernel = _variable_with_weight_decay('weights_1', shape=[1, 7, out_dim, out_dim], stddev=0.01, wd=0.0005)
        con_7 = tf.nn.conv2d(con_7, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)

        kernel = _variable_with_weight_decay('weights_2', shape=[7, 1, out_dim, out_dim], stddev=0.01, wd=0.0005)
        con_7 = tf.nn.conv2d(con_7, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_7)

    with tf.variable_scope('con_5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 5, in_dim, out_dim], stddev=0.01, wd=0.0005)
        con_5 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)

        kernel = _variable_with_weight_decay('weights_2', shape=[5, 1, out_dim, out_dim], stddev=0.01, wd=0.0005)
        con_5 = tf.nn.conv2d(con_5, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_5)

    with tf.variable_scope('con_3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, in_dim, out_dim], stddev=0.01, wd=0.0005)
        con_3 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_3)

    with tf.variable_scope('concat') as scope:
        concat = tf.concat([con_9, con_7, con_5, con_3], 3, name=scope.name)
        biases = _variable_on_cpu('biases', [out_dim * 4], tf.constant_initializer(0))
        bias = tf.nn.bias_add(concat, biases)

        if is_bn:
            bn = BatchNorm()
            bias = bn(bias)

        msb = tf.nn.relu(bias)

    return msb


def inference_bn(images):
    """
    Added batch normalization after the CNN model; modified the activation function: f(x)=relu(sigmoid(x))
    :in: original image
    :return: density approximation image
    """
    with tf.variable_scope('con1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=0.01, wd=0.0005)
        con = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        con1 = tf.nn.relu(bias)
        _activation_summary(con1)

    with tf.variable_scope('con2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=0.01, wd=0.0005)
        con = tf.nn.conv2d(con1, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        bias = tf.nn.bias_add(con, biases)
        con2 = tf.nn.relu(bias)
        _activation_summary(con2)

    with tf.variable_scope('con3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=0.01, wd=0.0005)
        con = tf.nn.conv2d(con2, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        bias = tf.nn.bias_add(con, biases)
        con3 = tf.nn.relu(bias)
        _activation_summary(con3)

    # msb_con2
    with tf.variable_scope('msb_con2'):
        msb_con2 = multi_scale_block(con3, 64, 16, is_bn=True)

    # pool_msb_con2
    with tf.variable_scope('pool_msb_con2') as scope:
        pool_msb_con2 = tf.nn.max_pool(msb_con2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con3
    with tf.variable_scope('msb_con3'):
        msb_con3 = multi_scale_block(pool_msb_con2, 64, 32, is_bn=True)
        _activation_summary(msb_con3)

    # msb_con4
    with tf.variable_scope('msb_con4'):
        msb_con4 = multi_scale_block(msb_con3, 128, 32, is_bn=True)
        _activation_summary(msb_con4)

   # pool_msb_con4
    with tf.variable_scope('pool_msb_con4') as scope:
        pool_msb_con4 = tf.nn.max_pool(msb_con4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con5
    with tf.variable_scope('msb_con5'):
        msb_con5 = multi_scale_block(pool_msb_con4, 128, 64, is_bn=True)
        _activation_summary(msb_con5)

    # msb_con6
    with tf.variable_scope('msb_con6'):
        msb_con6 = multi_scale_block(msb_con5, 256, 64, is_bn=True)
        _activation_summary(msb_con6)

    # short cut/residue connection
    with tf.variable_scope('short_cut'):
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 128, 256], stddev=0.01, wd=0.0005)
        short_con = tf.nn.conv2d(pool_msb_con4, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        short_con = msb_con6 + short_con
        _activation_summary(short_con)

    # msb_con7
    with tf.variable_scope('msb_con7'):
        msb_con7 = multi_scale_block(short_con, 256, 64, is_bn=True)
        _activation_summary(msb_con7)

    # final_conv
    with tf.variable_scope('final_conv') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, 1000], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(msb_con7, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        final_conv = tf.nn.relu(bias)
        _activation_summary(final_conv)

    # con_out
    with tf.variable_scope('con_out') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 1000, 1], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(final_conv, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)

        bn = BatchNorm()
        bias = bn(bias)

        con_out = tf.nn.relu(tf.nn.sigmoid(bias))
        _activation_summary(con_out)

    image_out = con_out

    tf.summary.image("con_img", image_out)

    return image_out


def loss(predict, label):
    """
    calculate loss
    :param predict: mscnn approximate density map
    :param label: ground truth crowd counting map
    :return: L1 smooth loss/L2 loss
    """
    # L1 smooth Loss
    predict = tf.squeeze(predict, 3)
    a = tf.abs(predict - label)
    # l1_smooth_loss = tf.reduce_sum(tf.where(tf.greater(1.0, a), 0.5*a*a, a-0.5))

    l2_loss = tf.reduce_sum((predict - label) * (predict - label))

    # add summary
    tf.summary.histogram('loss', l2_loss)

    return l2_loss


def add_avg_loss(avg_loss):
    """
    calculate average loss
    """
    add_avg_loss_op = avg_loss * 1
    tf.summary.histogram('avg_loss', avg_loss)

    return add_avg_loss_op


def train(total_loss, global_step, nums_per_train):
    # num_batches_per_epoch = nums_per_train / FLAGS.batch_size
    # decay_steps = int(num_batches_per_epoch * mscnn_train.num_epochs_per_decay)

    lr = tf.train.exponential_decay(mscnn_train.initial_learning_rate,
                                    global_step,
                                    1000,
                                    mscnn_train.learning_rate_per_decay,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    print(lr)

    # optimizer
    opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-8)
    grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    train_op = apply_gradient_op

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    return train_op
