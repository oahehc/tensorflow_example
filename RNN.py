"""
Reference:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime

# # DATASET
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# # CONFIG
epochs = 1001
print_range = (epochs - 1) // 20
n_hidden_units = 128
n_inputs = 28
n_steps = 28 # use image y-axis to simulate time
n_classes = mnist.train.labels.shape[1] # class 0~9 = 10
batch_size = 256
batch_num = mnist.train.images.shape[0] // batch_size


# # MODEL
input_x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
output_y = tf.placeholder(tf.float32, [None, n_classes])
# encoder layer: 28(x-axis) -> 128(feature)
encoder_weight = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
encoder_bias = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))
input_x_reshape = tf.reshape(input_x, [-1, n_inputs]) # (256 batch, 28 steps, 28) => (256 batch * 28, 28)
y1 = tf.matmul(input_x_reshape, encoder_weight) + encoder_bias # => (256 batch * 28, 128)
y1_reshape = tf.reshape(y1, [-1, n_steps, n_hidden_units]) # => (256 batch, 28 steps, 128)
# RNN:
# cell = tf.contrib.rnn.BasicRNNCell(n_hidden_units) # basic RNN
# cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units) # LSTM
# cell = tf.contrib.rnn.LSTMCell(n_hidden_units) # LSTM *able to set peephole, clip
cell = tf.contrib.rnn.GRUCell(n_hidden_units) # GRU
# state = cell.zero_state(batch_size, dtype=tf.float32)
output_rnn, final_state = tf.nn.dynamic_rnn(cell, y1_reshape, time_major=False, dtype=tf.float32) # time_major=False for data format = (batch, time, data)
# decoder layer: 128(feature) -> 10(class)
decoder_weight = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
decoder_bias = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
output_rnn_ = tf.unstack(tf.transpose(output_rnn, perm=[1, 0, 2])) # transpose to (steps(time) * batch * data)
prediction = tf.matmul(output_rnn_[-1], decoder_weight) + decoder_bias # use last output from RNN
# accuracy:
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(output_y, 1)), tf.float32))
# loss:
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_y))
# optimizer: Adam
train_op = tf.train.AdamOptimizer().minimize(cost)


# # TRAIN
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            input_x: batch_xs,
            output_y: batch_ys,
        })
        # print train process
        if epoch%print_range == 0:
            accuracy_ = sess.run(accuracy, feed_dict={
                input_x: batch_xs,
                output_y: batch_ys,
            })
            print(datetime.now(), epoch, accuracy_)
    print('*** finish ***')
    
