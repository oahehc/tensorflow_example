'''
reference: 
https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/14_DCGAN_with_MNIST.ipynb
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

# # DATASET
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # read mnist data


# # DEFINE FUNCTION
def weight_variable(shape, name='var'):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name)
def bias_variable(shape, name='var'):
    return tf.Variable(tf.zeros(shape) + 0.1, name)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'SAME')
def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')
def sample(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap = 'gray')
    plt.show()


# # CONFIG
epochs = 11
print_range = (epochs - 1) // 10
batch_size = 256
batch_num = mnist.train.images.shape[0] // batch_size
input_depth = 1
input_width = 28
filter_size = 5
cv1_dim = 32
cv2_dim = 64
output_depth = 1
output_width = input_width//2//2
input_generator = 128
# 28*28*1 > 14*14*32(strides=2) > 7*7*64(strides=2) > 1
dis = {
    'layer1_w': [filter_size, filter_size, input_depth, cv1_dim], # convolution
    'layer1_b': [cv1_dim],
    'layer2_w': [filter_size, filter_size, cv1_dim, cv2_dim], # convolution
    'layer2_b': [cv2_dim],
    'layer3_w': [output_width * output_width * cv2_dim, output_depth], # fully connected
    'layer3_b': [output_depth],
}
# 128 > 7*7*64 > 14*14*32 > 28*28*1
gen = {
    'layer1_w': [input_generator, output_width * output_width * cv2_dim], # fully connected
    'layer1_b': [output_width * output_width * cv2_dim],
    'layer2_w': [filter_size, filter_size, cv1_dim, cv2_dim], # deconvolution
    'layer2_b': [cv1_dim],
    'layer3_w': [filter_size, filter_size, input_depth, cv1_dim], # deconvolution
    'layer3_b': [input_depth],
}

# # MODEL
input_gen = tf.placeholder(tf.float32, shape = [None, input_generator])
W1_gen = weight_variable(gen['layer1_w'])
b1_gen = bias_variable(gen['layer1_b'])
W2_gen = weight_variable(gen['layer2_w'])
b2_gen = bias_variable(gen['layer2_b'])
W3_gen = weight_variable(gen['layer3_w'])
b3_gen = bias_variable(gen['layer3_b'])
var_gen = [W1_gen, b1_gen, W2_gen, b2_gen, W3_gen, b3_gen]
def generator(inputData):
    input_size = tf.shape(inputData)[0]
    h1_gen = tf.nn.relu(tf.add(tf.matmul(inputData, W1_gen), b1_gen))
    h1_gen_reshape = tf.reshape(h1_gen, [-1, output_width, output_width, cv2_dim])
    output_shape_g2 = tf.stack([input_size, input_width//2, input_width//2, cv1_dim])
    h2_gen = tf.nn.relu(tf.add(deconv2d(h1_gen_reshape, W2_gen, output_shape_g2), b2_gen))
    output_shape_g3 = tf.stack([input_size, input_width, input_width, input_depth])
    output_gen = tf.nn.tanh(tf.add(deconv2d(h2_gen, W3_gen, output_shape_g3), b3_gen))
    return output_gen

input_dis = tf.placeholder(tf.float32, [None, input_width*input_width*input_depth])
W1_dis = weight_variable(dis['layer1_w'])
b1_dis = bias_variable(dis['layer1_b'])
W2_dis = weight_variable(dis['layer2_w'])
b2_dis = bias_variable(dis['layer2_b'])
W3_dis = weight_variable(dis['layer3_w'])
b3_dis = bias_variable(dis['layer3_b'])
var_dis = [W1_dis, b1_dis, W2_dis, b2_dis, W3_dis, b3_dis]
def discriminator(inputData):
    input_reshape = tf.reshape(inputData, [-1, input_width, input_width, input_depth])
    h1_dis = tf.nn.relu(tf.add(conv2d(input_reshape, W1_dis), b1_dis))
    h2_dis = tf.nn.relu(tf.add(conv2d(h1_dis, W2_dis), b2_dis))
    h2_dis_reshape = tf.reshape(h2_dis, [-1, output_width * output_width * cv2_dim])
    output_dis = tf.nn.sigmoid(tf.add(tf.matmul(h2_dis_reshape, W3_dis), b3_dis))
    return output_dis


sample_gen = generator(input_gen)
dis_real = discriminator(input_dis)
dis_fake = discriminator(sample_gen)
# loss
loss_dis = -tf.reduce_mean(tf.log(dis_real) + tf.log(1.0 - dis_fake))
loss_gen = -tf.reduce_mean(tf.log(dis_fake))
optimizer_dis = tf.train.AdamOptimizer(0.0001).minimize(loss_dis, var_list= var_dis)
optimizer_gen = tf.train.AdamOptimizer(0.0001).minimize(loss_gen, var_list= var_gen)


# # TRAIN
min_loss = 0.001 # early stop if loss goes to 0
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        for i in range(batch_num):
            batch_x = mnist.train.next_batch(batch_size)[0]
            _, loss_dis_train = sess.run([optimizer_dis, loss_dis], 
                feed_dict = {input_dis: batch_x, input_gen: sample(batch_size, input_generator)})
            _, loss_gen_train = sess.run([optimizer_gen, loss_gen], 
                feed_dict = {input_gen: sample(batch_size, input_generator)})
        if loss_dis_train < min_loss or math.isnan(loss_dis_train):
            print('early stop ', epoch, loss_dis_train, loss_gen_train)
            break
        elif epoch%print_range == 0:
            print(datetime.now(), epoch, loss_dis_train, loss_gen_train)
    sample_image = sample_gen.eval(feed_dict = {input_gen: sample(16, input_generator)})
    plot(sample_image)
    print('*** finish ***')



