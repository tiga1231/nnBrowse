import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

dataFn = 'data_c/data.js'
try:
    os.remove(dataFn)
except:
    pass
with open(dataFn, 'a') as f:
    f.write('var dd = {};\n')


def dump(obj, varName):
    with open(dataFn, 'a') as f:
        f.write('dd.'+varName +'=\n')
        json.dump(obj, f)
        f.write(';\n')


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

m = {}
for i,d in enumerate([10,]*30):
    for _ in range(d):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    wi = sess.run(W).T
#a = mnist.train.images[0]
    m['layer0_t'+str(i)] = wi.tolist()
dump(m, 'W')
dump(i+1, 'stepCount')

xs, yTrue = mnist.train.next_batch(200)
yTrueLabel = np.argmax(yTrue, axis=1)
yPred = sess.run(y, feed_dict={x: xs})
yPredLabel = np.argmax(yPred, axis=1)
dump(xs.tolist(), 'x')
dump(yTrueLabel.tolist(), 'labelTrue')
dump(yPredLabel.tolist(), 'labelPred')
#TODO make yPred vary in time
dump(yPred.tolist(), 'y')
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

