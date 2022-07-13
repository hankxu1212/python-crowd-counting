import time
import numpy as np
import cv2
from tensorflow.python.platform import gfile
import tensorflow as tf
from densityDetection import mscnn_improved_3 as mscnn
import warnings

# variables
eval_dir = 'eval'
data_test_gt = 'Data_original/Data_gt/test_gt/'
data_test_im = 'Data_original/Data_im/test_im/'
data_test_index = 'Data_original/dir_name_test.txt'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', eval_dir, """日志目录""")
tf.app.flags.DEFINE_string('data_test_gt', data_test_gt, """测试集集标签""")
tf.app.flags.DEFINE_string('data_test_im', data_test_im, """测试集图片""")
tf.app.flags.DEFINE_string('data_test_index', data_test_index, """测试集图片""")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def evaluate():
    """
    on ShanghaiTech data set eval
    """
    images = tf.placeholder("float")
    labels = tf.placeholder("float")

    #predict_op = mscnn.inference(images)
    predict_op = mscnn.inference_bn(images)
    loss_op = mscnn.loss(predict_op, labels)

    # load model parameters
    saver = tf.train.Saver()
    sess = tf.Session()

    # initialize and create session
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    checkpoint_dir = tf.train.get_checkpoint_state(FLAGS.model_dir)
    saver.restore(sess, 'C:/Users/admin\PycharmProjects\AI\densityDetection\model\skip_mcnn.ckpt-1500')
    dir_file = open(FLAGS.data_test_index)
    dir_names = dir_file.readlines()

    step = 0
    sum_all_mae = 0
    sum_all_mse = 0
    for file_name in dir_names:
        step += 1
        im_name, gt_name = file_name.split(' ')
        gt_name = gt_name.split('\n')[0]

        batch_xs = cv2.imread(FLAGS.data_test_im + im_name)
        batch_xs = np.array(batch_xs, dtype=np.float32)
        batch_xs = batch_xs.reshape(1, len(batch_xs), -1, 3)

        # data (density map)
        batch_ys = np.array(np.load(FLAGS.data_test_gt + gt_name))
        batch_ys = np.array(batch_ys, dtype=np.float32)
        batch_ys = batch_ys.reshape(1, len(batch_ys), -1)

        start = time.process_time()
        predict = sess.run([predict_op], feed_dict={images: batch_xs})
        loss_value = sess.run(loss_op, feed_dict={images: batch_xs, labels: batch_ys})
        end = time.process_time()

        print("time: %s\t loss_value: %s\t counting:%.7f\t predict:%.7f\t diff:%.7f" % \
              ((end - start), loss_value, sum(sum(sum(batch_ys))), sum(sum(sum(predict[0]))),
               sum(sum(sum(batch_ys)))-sum(sum(sum(predict[0])))))

        sum_ab = abs(sum(sum(sum(batch_ys))) - sum(sum(sum(predict[0]))))
        sum_all_mae += sum_ab
        sum_all_mse += sum_ab * sum_ab

    avg_mae = sum_all_mae / len(dir_names)
    avg_mse = (sum_all_mse / len(dir_names)) ** 0.5
    print("MAE: %.7f\t MSE:%.7f" % (avg_mae, avg_mse))


def main(argv=None):
    if gfile.Exists(FLAGS.eval_dir):
        gfile.DeleteRecursively(FLAGS.eval_dir)
    gfile.MakeDirs(FLAGS.eval_dir)

    evaluate()

if __name__ == '__main__':
    tf.app.run()
