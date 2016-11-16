import cPickle as pickle
import tensorflow as tf
import os.path
import sys
sys.path.append('./config/')
import base as base

weight_file_name = 'nn1024_weights.pkl'
log_file_name = 'nn1024.log'

def create_weights(number_input, number_output, logging = True):
    weights = {}
    if os.path.isfile(weight_file_name):
        if logging:
            print "Weight File Found! loading weights!"
        data = pickle.load( open(weight_file_name, "rb" ) )
        for key, val in data.iteritems():
            weights[key] = tf.Variable(val)
    else:
        if logging:
            print "Initilizing new weights."
        weights = {
            'w_1': base.weight_variable([number_input,1024]),
            'b_1': base.bias_variable([1024]),

            'w_2': base.weight_variable([1024,number_output]),
            'b_2': base.bias_variable([number_output])
        }
    return weights

def create_network(x, weights, keep_prob):
    h_1 = tf.nn.relu(tf.matmul(x, weights['w_1']) + weights['b_1'])
    h_drop = tf.nn.dropout(h_1, keep_prob)
    return tf.nn.softmax(tf.matmul(h_drop, weights['w_2']) + weights['b_2'])
