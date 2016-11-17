import numpy as np
import matplotlib
import tensorflow as tf
import sys
import pickle
from network import Network
from logger import Logger


class ASDS():
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.networks = ['nn256', 'nn1024', 'nn256_256', 'nn1024_256', 'nn1024_1024', 'nn256_256_256']
        self.logger = Logger(shouldLog=True)

    def train(self, TrainingData):
        for net in self.networks:
            self.logger.update(net)
            data = TrainingData(self.logger)
            sys.stdout.flush()
            numberOuput = data.get_num_out()
            numberInput = data.get_num_in()

            x = tf.placeholder("float", shape=[None, numberInput])
            y_ = tf.placeholder("float", shape=[None, numberOuput])
            keep_prob = tf.placeholder(tf.float32)
            network = Network(x, y_, numberOuput, numberInput, keep_prob, net, self.logger)
            y_conv = network.create_network()
            cross_entropy = network.get_loss()

            # Exponetial Decaying learning rate to better learn down the road.
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 1e-5
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)

            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.sess.run(tf.initialize_all_variables())

            batch_size = 10
            num_iter = 5
            prob = 0.8
            logging = 1
            self.logger.log('Running network %d times on batches of %d logging every %d and training on %f of the data on %s network' %(num_iter,batch_size,logging, prob,net))

            # start train loop
            for i in range(num_iter):
                batch = data.get_batch(batch_size)
                if len(batch[0]) < 1:
                    continue
                if i%logging == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                    self.logger.log("\nstep %d, training accuracy %g"%(i, train_accuracy))
                    # print ("\nstep %d, training accuracy %g"%(i, train_accuracy))
                    predict_y = y_conv.eval(feed_dict={x:batch[0], keep_prob: 1.0})
                    # cf(batch[1],predict_y, number_output)
                else:
                    print '\rIteration {0}'.format(i),
                    sys.stdout.flush()
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: prob})
            print ''
            # sum_perc = 0
            # total_perc = 0
            # while not data.is_at_test_batch_end():
            #     batch = data.get_test_batch()
            #     sum_perc += accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            #     total_perc += 1
            # print('')
            # print("Accuracy %g"%(sum_perc / total_perc))
            network.save()

from data_person import TrainingData
asds = ASDS()

asds.train(TrainingData)
