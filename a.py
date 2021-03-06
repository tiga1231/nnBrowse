import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def dump(obj, varName):
    with open('data/'+varName +'.js', 'w') as f:
        f.write('var '+varName +'=\n')
        json.dump(obj, f)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#a = mnist.train.images[0]
#imshow(a)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


bb_xs, bb_ys = mnist.test.next_batch(120)
yTrue = np.argmax(bb_ys, axis=1)
dump(bb_xs.tolist(), 'x')
dump(yTrue.tolist(), 'yTrueLabel')


kl = 500
perplexity = 20
trialCount = 20
for i in range(trialCount):
    model = TSNE(perplexity=perplexity)
    xProj_i = model.fit_transform(bb_xs)
    if model.kl_divergence_ < kl:
        kl = model.kl_divergence_
        xProj = xProj_i
        print i, model.kl_divergence_
    else:
        print i
#model = PCA()
#xProj = model.fit_transform(bb_xs)
dump(xProj.tolist(), 'xProj')

yProjs = []
yPredLabels = []
for sub,r in enumerate([500,500]):
    for _ in range(r):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print sess.run( accuracy,
    feed_dict={ x: mnist.test.images, y_: mnist.test.labels})

    #viz y
    yPred = sess.run(y, feed_dict={x: bb_xs, y_: bb_ys})
    yPredLabel = np.argmax(yPred, axis=1)
    yPredLabels.append(yPredLabel.tolist())

    kl = 500
    for i in range(trialCount):
        model = TSNE(perplexity=perplexity)
        yProj_i = model.fit_transform(yPred)
        if model.kl_divergence_ < kl:
            kl = model.kl_divergence_
            yProj = yProj_i
            print i, model.kl_divergence_
        else:
            print i
    #model = PCA()
    #yProj = model.fit_transform(yPred)
    yProjs.append(yProj.tolist())
dump(yProjs, 'yProjs')
dump(yPredLabels, 'yPredLabels')
#plt.show()
