import numpy as np
import matplotlib
import tensorflow as tf
import sys
import pickle
from data_both import TrainingData
from network import Network
data = TrainingData()
sys.stdout.flush()
numberOuput = data.get_num_out()
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 2000]) #100x20
y_ = tf.placeholder("float", shape=[None, numberOuput])
keep_prob = tf.placeholder(tf.float32)
network = Network(x, y_, numberOuput, keep_prob)
y_conv = network.create_network()
cross_entropy = network.get_loss()

train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

batch_size = 10
num_iter = 100000000
prob = 0.8
logging = 1000
print 'Running network %d times on batches of %d logging every %d and training on %f of the data' %(num_iter,batch_size,logging, prob)

# start train loop, to exit hit ctrl-c
try:
    for i in range(num_iter):
        batch = data.get_batch(batch_size)
        if i%logging == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("\nstep %d, training accuracy %g"%(i, train_accuracy))
            predict_y = y_conv.eval(feed_dict={x:batch[0], keep_prob: 1.0})
            # cf(batch[1],predict_y, number_output)
        else:
            print '\rIteration {0}'.format(i),
            sys.stdout.flush()
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: prob})
except KeyboardInterrupt:
    sum_perc = 0
    total_perc = 0
    while not data.is_at_test_batch_end():
        batch = data.get_test_batch()
        sum_perc += accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        total_perc += 1
    print('')
    print("Accuracy %g"%(sum_perc / total_perc))
    network.save()
