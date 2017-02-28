import tensorflow as tf
from PIL import Image
from os import listdir
from random import shuffle
import numpy as np
import pickle


photos = []
# change the directory here to the path of the folder for the images you want to classify
cereal_dir = 'arielle/'
cereal_photos = [f for f in listdir(cereal_dir)]

print("Saving Arielle")

for i, s in enumerate(cereal_photos):
    try:
        im = Image.open(cereal_dir + s).convert()
    except:
        continue
    photos.append(np.asarray(im.getdata(), dtype=np.float64))



sess = tf.Session()
new_saver = tf.train.import_meta_graph('CNN.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.trainable_variables()

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],\
        strides=[1,2,2,1],padding='SAME')
def max_pool_5x5(x):
    return tf.nn.max_pool(x,ksize=[1,5,5,1],\
        strides=[1,5,5,1],padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 10000,3])

W_conv1 = all_vars[0]
b_conv1 = all_vars[1]

x_image = tf.reshape(x,[-1,100,100,3])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = all_vars[2]
b_conv2 = all_vars[3]

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = all_vars[4]
b_conv3 = all_vars[5]

h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
h_pool3 = max_pool_5x5(h_conv3)

W_fc1 = all_vars[6]
b_fc1 = all_vars[7]

h_pool3_flat = tf.reshape(h_pool3,[-1,5*5*160])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = all_vars[8]
b_fc2 = all_vars[9]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

correct_prediction = tf.equal(tf.argmax(y_conv,1), 1)
soup = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


percent = soup.eval(feed_dict = {\
    x: photos,keep_prob:1.0},session=sess)

percent = (percent - .16)/(.84 - .16) * 100

print("Arielle is %g percent soup"%percent)

