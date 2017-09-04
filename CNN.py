import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import matplotlib.pyplot as plt

# # DEFINE FUNCTION
def activationFunc(inputData, activation=None):
    if activation:
        return activation(inputData)
    else:
        return inputData
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape) + 0.1)


# # DATASET
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # read mnist data


# # CONFIG
epochs = 11
print_range = (epochs - 1) // 10
inputDimension = mnist.train.images.shape[1] # 28*28 = 784
outputDimension = mnist.train.labels.shape[1] # class 0~9 = 10
batch_size = 512
batch_num = mnist.train.images.shape[0] // batch_size
train_drop_rate = 0.5
layer1_FilterSize = 5
layer1_Depth = 4
layer2_FilterSize = 5
layer2_Depth = 8
layer3 = 512
activation = tf.nn.relu


# # MODEL
# 28*28*1 > 14*14*4 > 7*7*8 > 512 > 10
input_x = tf.placeholder(tf.float32, [None, inputDimension])
input_x_2d = tf.reshape(input_x, [-1, 28, 28, 1]) # reshape to 2-dimension
drop_rate = tf.placeholder(tf.float32)
# layer1 conv : sample padding 5*5 2d-convolution + 2*2 maxpool
W1_conv = weight_variable([layer1_FilterSize, layer1_FilterSize, 1, layer1_Depth])
b1_conv = bias_variable([layer1_Depth])
h1_conv = activationFunc(tf.nn.conv2d(input_x_2d, W1_conv, strides=[1, 1, 1, 1], padding='SAME') + b1_conv, activation)
y1_conv = tf.nn.max_pool(h1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# layer2 conv : sample padding 5*5 2d-convolution + 2*2 maxpool
W2_conv = weight_variable([layer2_FilterSize, layer2_FilterSize, layer1_Depth, layer2_Depth])
b2_conv = bias_variable([layer2_Depth])
h2_conv = activationFunc(tf.nn.conv2d(y1_conv, W2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv, activation)
y2_conv = tf.nn.max_pool(h2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
y2_conv_flat = tf.reshape(y2_conv, [-1, 7*7*layer2_Depth])
# layer3 full connect : 
W3_fc = weight_variable([7*7*layer2_Depth, layer3])
b3_fc = bias_variable([layer3])
y3_fc = activationFunc(tf.matmul(y2_conv_flat, tf.nn.dropout(W3_fc, drop_rate)) + b3_fc, activation)
# output layer : 
W4_fc = weight_variable([layer3, outputDimension])
b4_fc = bias_variable([outputDimension])
prediction = tf.nn.softmax(tf.matmul(y3_fc, W4_fc) + b4_fc)

# loss: use cross_entropy to calculate distance between two distribution
output_y = tf.placeholder(tf.float32, [None, outputDimension])
loss = tf.reduce_mean(-tf.reduce_sum(output_y*tf.log(prediction), axis=1))
# accuracy: choose max probability as prediction
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(output_y,1)), tf.float32))
# optimizer: Adam
train_step = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
    # # TRAIN
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        # min batch
        for i in range(batch_num):
            batch_data, batch_label = mnist.train.next_batch(batch_size)   
            sess.run(train_step, feed_dict={input_x:batch_data, output_y:batch_label, drop_rate: train_drop_rate})

        # print train process
        if epoch%print_range == 0:
            # use whole train dataset in calculate accuracy might exhaust memory
            accuracy_train = sess.run(accuracy, feed_dict={input_x:batch_data, output_y:batch_label, drop_rate: 1.0})
            accuracy_test = sess.run(accuracy, feed_dict={input_x:mnist.test.images, output_y:mnist.test.labels, drop_rate: 1.0})
            print(datetime.now(), epoch)
            print('- Train: ', accuracy_train)
            print('-  Test: ', accuracy_test)
    print('*** finish ***')

    # # TRAINING RESULT TEST
    test_dataset = mnist.test.images
    test_image = test_dataset[0:9, ]
    test_result = sess.run(tf.argmax(prediction,1), feed_dict={input_x:test_image, drop_rate: 1.0})
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.reshape(test_image[i], (28, 28)))
        xlabel = "Pred: {0}".format(test_result[i])
        ax.set_xlabel(xlabel)
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()