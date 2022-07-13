import os.path
import random
import cv2
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from densityDetection import mscnn_improved_4 as mscnn
import warnings

# model variables
FLAGS = tf.app.flags.FLAGS

# model training variables
num_epochs_per_decay = 20
learning_rate_per_decay = 0.90
initial_learning_rate = 1.0e-1
warnings.filterwarnings("ignore", category=DeprecationWarning)

def train():
    """
    using ShanghaiTech database to train
    """
    with tf.Graph().as_default():
        # read directory .txt
        dir_file = open(FLAGS.data_train_index)
        # get data file name and target file name
        dir_name = dir_file.readlines()
        # parameter settings
        nums_train = len(dir_name)  # # of images per train
        global_step = tf.Variable(0, trainable=False)  # define global decay steps

        # place_holder
        image = tf.compat.v1.placeholder("float")
        label = tf.compat.v1.placeholder("float")
        avg_loss = tf.compat.v1.placeholder("float")

        # related initializations
        predicts = mscnn.inference_bn(image)  # create model
        loss = mscnn.loss(predicts, label)  # calculate loss
        train_op = mscnn.train(loss, global_step, nums_train)  # train op

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))  # create a session
        saver = tf.train.Saver(tf.global_variables())  # saver

        init = tf.initialize_all_variables()  # initialization
        sess.run(init)  # initialize all variables

        checkpoint_dir = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if checkpoint_dir and checkpoint_dir.model_checkpoint_path:
            saver.restore(sess, checkpoint_dir.model_checkpoint_path)
        else:
            print('Not found checkpoint file')

        summary_op = tf.summary.merge_all()  # summary
        add_avg_loss_op = mscnn.add_avg_loss(avg_loss)  # add ave_loss's operation
        summary_writer = tf.summary.FileWriter(FLAGS.train_log, graph_def=sess.graph_def)  # create a summarywriter

        steps = 2001
        avg_loss_1 = 0

        for step in xrange(steps):
            # if step < nums_train * 10:
            #     # start 10 times iteration in order of epochs of 700
            #     num_batch = [divmod(step, nums_train)[1] + i for i in range(FLAGS.Batch_size)]
            # else:
            num_batch = random.sample(range(nums_train), nums_train)[0:FLAGS.batch_size]
            xs, ys = [], []
            for index in num_batch:

                # get directories
                file_name = dir_name[index]
                im_name, gt_name = file_name.split(' ')
                gt_name = gt_name.split('\n')[0]

                # training data (image)
                batch_xs = cv2.imread(FLAGS.data_train_im + im_name)
                batch_xs = np.array(batch_xs, dtype=np.float32)

                # training data (density map)
                batch_ys = np.array(np.load(FLAGS.data_train_gt + gt_name))
                batch_ys = np.array(batch_ys, dtype=np.float32)
                batch_ys = batch_ys.reshape([batch_ys.shape[0], batch_ys.shape[1], -1])

                xs.append(batch_xs)
                ys.append(batch_ys)

            np_xs = np.array(xs)
            np_ys = np.array(ys)[:,:,:,0]

            # get loss and label density map
            _, loss_value = sess.run([train_op, loss], feed_dict={image: np_xs, label: np_ys})
            output = sess.run(predicts, feed_dict={image: np_xs})
            avg_loss_1 += loss_value

            # sumamry
            # if step % 5 == 0:
            #     summary_str = sess.run(summary_op, feed_dict={image: np_xs, label: np_ys, avg_loss: avg_loss_1 / 5})
            #     summary_writer.add_summary(summary_str, step)
            #     avg_loss_1 = 0

            if step % 1 == 0:
                ground_truth = sum(sum(sum(np_ys)))
                output_density = sum(sum(sum(output)))
                print("step:%d avg_loss:%.5f\t counting:%.5f\t predict:%.5f" % \
                      (step, loss_value, ground_truth, output_density))

                # write into files
                # f = open('C:/Users/admin\Desktop/trainingdensity/10/log10_loss.txt', 'a')
                # f.write(str(loss_value) + "\n")
                # f.close()
                # f2 = open('C:/Users/admin\Desktop/trainingdensity/10/log10_difference_graph.txt', 'a')
                # percentage = abs((ground_truth-output_density)/output_density)
                # f2.write(str(percentage[0]) + "\n")
                # f2.close()
                # tf.summary.histogram('percent', percentage)
                sess.run(add_avg_loss_op, feed_dict={avg_loss: loss_value})

            # save model parameters
            if step % 500 == 0 and step != 0:
                checkpoint_path = os.path.join(FLAGS.model_dir, 'skip_mcnn.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            # output predicted density map
            if step % 1000 == 0 and step != 0:
                out_path = os.path.join(FLAGS.output_dir, str(step) + "out.npy")
                np.save(out_path, output)


def main(argv=None):
    if gfile.Exists(FLAGS.train_log):
        gfile.DeleteRecursively(FLAGS.train_log)
    gfile.MakeDirs(FLAGS.train_log)

    if not gfile.Exists(FLAGS.model_dir):
        gfile.MakeDirs(FLAGS.model_dir)

    if not gfile.Exists(FLAGS.output_dir):
        gfile.MakeDirs(FLAGS.output_dir)

    train()


if __name__ == '__main__':
    tf.compat.v1.app.run()
