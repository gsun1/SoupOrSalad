from PIL import Image
from os import listdir
from random import shuffle
import numpy as np
import pickle
import tensorflow as tf


train_size = 0.8
batch_size = 10

soup_dir = "soup/"
salad_dir = "salad/"
soup_photos = [f for f in listdir(soup_dir)]
salad_photos = [f for f in listdir(salad_dir)]
shuffle(soup_photos)
shuffle(salad_photos)

train_images = []
train_classes = []

test_images = []
test_classes = []
train = []
test = []

# 1 is soup, 0 is salad
print("Saving soups")
for i, s in enumerate(soup_photos):
    try:
        im = Image.open(soup_dir + s).convert()
    except:
        continue
    y = np.asarray(im.getdata(), dtype=np.float64)
    if i < int(len(soup_photos)*train_size):
        train.append((y,1))
        #train_images.append(y)
        #train_classes.append(1)
    else:
        test.append((y,1))
        #test_images.append(y)
        #test_classes.append(1)
print("Saving salads")
for i, s in enumerate(salad_photos):
    try:
        im = Image.open(salad_dir + s).convert()
    except:
        continue
    y = np.asarray(im.getdata(), dtype=np.float64)
    if i < int(len(salad_photos)*train_size):
        train.append((y,0))
        #train_images.append(y)
        #train_classes.append(0)
    else:
        test.append((y,0))
        #test_images.append(y)
        #test_classes.append(0)

# shuffle everything around
shuffle(train)
shuffle(test)
for i in train:
    if i[1] == 0:
        train_classes.append([1,0])
    else:
        train_classes.append([0,1])
    train_images.append(i[0])
for i in test:
    if i[1] == 0:
        test_classes.append([1,0])
    else:
        test_classes.append([0,1])
    test_images.append(i[0])

def batch(lst1,lst2,num):
    assert(len(lst1) == len(lst2))
    num_batches = len(lst1) // num
    result = []
    for i in range(num_batches):
        result.append((lst1[i*num:(i+1)*num],lst2[i*num:(i+1)*num]))
    return result

batches = batch(train_images,train_classes,batch_size)
#print(batches[0])


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],\
        strides=[1,2,2,1],padding='SAME')
def max_pool_5x5(x):
    return tf.nn.max_pool(x,ksize=[1,5,5,1],\
        strides=[1,5,5,1],padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 10000,3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W_conv1 = weight_variable([5,5,3,16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(x,[-1,100,100,3])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3,3,16,32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3,3,32,160])
b_conv3 = bias_variable([160])

h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
h_pool3 = max_pool_5x5(h_conv3)

W_fc1 = weight_variable([5*5*160,1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3,[-1,5*5*160])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print("Convolving nets")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
count = 1
threshold = 200
for batch in batches:
    train_step.run(feed_dict={x:batch[0], y_: batch[1], keep_prob:0.5},session=sess)
    if count * batch_size >= threshold:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0},session=sess)
        print("batch %d, training accuracy %g"%(count, train_accuracy))
        threshold += 200
    count += 1


print("test accuracy %g"%accuracy.eval(feed_dict = {\
    x: test_images, y_: test_classes,keep_prob:1.0},session=sess))


#saver = tf.train.Saver()
#saver.save(sess,save_path='./CNN')