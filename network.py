import os.path
import numpy as np
from PIL import Image
import tensorflow as tf
import cPickle as pickle

class Network():
    def __init__(self, x, y_, number_output, keep_prob, file_name = 'weights.pkl', logging = True):
        self.x = x
        self.y_ = y_
        self.keep_prob = keep_prob
        self.file_name = file_name
        self.logging = logging

        self.create_weights(number_output)

    def create_weights(self, number_output):
        if os.path.isfile(self.file_name):
            if self.logging:
                print "Weight File Found! loading weights!"
            weights = pickle.load( open( self.file_name, "rb" ) )
            self.weights = {}
            for key, val in weights.iteritems():
                self.weights[key] = tf.Variable(val)
        else:
            if self.logging:
                print "Initilizing new weights."
            self.weights = {
                'w_1':self.weight_variable([2000,2048]),
                'b_1':self.bias_variable([2048]),

                'w_2':self.weight_variable([2048,2048]),
                'b_2':self.bias_variable([2048]),

                'w_3':self.weight_variable([2048,1024]),
                'b_3':self.bias_variable([1024]),

                'w_4':self.weight_variable([1024,number_output]),
                'b_4':self.bias_variable([number_output])
            }

    def create_network(self):
        h_1 = tf.nn.relu(tf.matmul(self.x, self.weights['w_1']) + self.weights['b_1'])
        # h_pool1 = self.max_pool_2x2(self.h_conv1) #50x10

        h_2 = tf.nn.relu(tf.matmul(h_1, self.weights['w_2']) + self.weights['b_2'])
        # h_pool2 = self.max_pool_2x2(self.h_conv2) #25x5

        h_3 = tf.nn.relu(tf.matmul(h_2, self.weights['w_3']) + self.weights['b_3'])

        h_drop = tf.nn.dropout(h_3, self.keep_prob)

        self.network = tf.nn.softmax(tf.matmul(h_drop, self.weights['w_4']) + self.weights['b_4'])
        return self.network

    def get_loss(self):
        return tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.network), reduction_indices=[1]))

    def get_regularization(self):
        return (tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.b_fc1) +
        tf.nn.l2_loss(self.W_fc2) + tf.nn.l2_loss(self.b_fc2))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(self, x, W, strides=[1, 1, 1, 1], padding='SAME'):
        return tf.nn.conv2d(x, W, strides=strides, padding=padding)

    def max_pool_2x2(self, x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)


    def save(self):
        if self.logging:
            print "Saving Weights"
        output = open(self.file_name, 'wb')
        pickle.dump(self.weights, output)
        if self.logging:
            print "Weights Saved!"
