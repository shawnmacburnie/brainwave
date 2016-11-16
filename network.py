import os.path
import numpy as np
from PIL import Image
import tensorflow as tf
import cPickle as pickle
import config.loader as loader

class Network():
    def __init__(self, x, y_, number_output, number_input, keep_prob, netType, file_name = 'weights.pkl', logging = True):
        self.x = x
        self.y_ = y_
        self.keep_prob = keep_prob
        self.file_name = file_name
        self.logging = logging
        self.nn = loader.load(netType)

        self.create_weights(number_input, number_output)

    def create_weights(self, number_input, number_output):
        self.weights = self.nn.create_weights(number_input, number_output, self.file_name, self.logging)

    def create_network(self):
        self.network = self.nn.create_network(self.x, self.weights, self.keep_prob)
        return self.network

    def get_loss(self):
        return tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.network), reduction_indices=[1]))

    def get_regularization(self):
        return (tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.b_fc1) +
        tf.nn.l2_loss(self.W_fc2) + tf.nn.l2_loss(self.b_fc2))


    def save(self):
        if self.logging:
            print "Saving Weights"
        weights = {}
        for key, val in self.weights.iteritems():
            weights[key] = val.eval()
        output = open(self.file_name, 'wb')
        pickle.dump(weights, output)
        if self.logging:
            print "Weights Saved!"
