'''
reference: 
https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/13_Generative_Adversarial_Network.ipynb
https://github.com/soumith/ganhacks

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
epochs = 501
print_range = (epochs - 1) // 10
batch_size = 256
batch_num = mnist.train.images.shape[0] // batch_size
gen = {
    'input_dim': 128,
    'layer1_dim': 256,
    'output_dim': mnist.train.images.shape[1] # 28*28 = 784
}
dis = {
    'input_dim': mnist.train.images.shape[1], # 28*28 = 784
    'layer1_dim': 256,
    'output_dim': 1
}


# # MODEL
drop_rate = tf.placeholder(tf.float32)
input_gen = tf.placeholder(tf.float32, shape = [None, gen['input_dim']])
W1_gen = weight_variable([gen['input_dim'], gen['layer1_dim']])
b1_gen = bias_variable([gen['layer1_dim']])
W2_gen = weight_variable([gen['layer1_dim'], gen['output_dim']])
b2_gen = bias_variable([gen['output_dim']])
var_gen = [W1_gen, b1_gen, W2_gen, b2_gen]
def generator(inputData, drop_rate):
    # dropout for generator to create noise
    h1_gen = tf.nn.softplus(tf.matmul(inputData, tf.nn.dropout(W1_gen, drop_rate)) + b1_gen)
    output_gen = tf.nn.sigmoid(tf.matmul(h1_gen, tf.nn.dropout(W2_gen, drop_rate)) + b2_gen)
    return output_gen

input_dis = tf.placeholder(tf.float32, shape = [None, dis['input_dim']])
W1_dis = weight_variable([dis['input_dim'], dis['layer1_dim']])
b1_dis = bias_variable([dis['layer1_dim']])
W2_dis = weight_variable([dis['layer1_dim'], dis['output_dim']])
b2_dis = bias_variable([dis['output_dim']])
var_dis = [W1_dis, b1_dis, W2_dis, b2_dis]
def discriminator(inputData):
    h1_dis = tf.nn.softplus(tf.matmul(inputData, W1_dis) + b1_dis)
    output_dis = tf.nn.sigmoid(tf.matmul(h1_dis, W2_dis) + b2_dis)
    return output_dis

sample_gen = generator(input_gen, drop_rate)
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
                feed_dict = {input_dis: batch_x, input_gen: sample(batch_size, gen['input_dim']), drop_rate: 0.5})
            _, loss_gen_train = sess.run([optimizer_gen, loss_gen], 
                feed_dict = {input_gen: sample(batch_size, gen['input_dim']), drop_rate: 0.5})
        if loss_dis_train < min_loss or math.isnan(loss_dis_train):
            print('early stop ', epoch, loss_dis_train, loss_gen_train)
            break
        elif epoch%print_range == 0:
            print(datetime.now(), epoch, loss_dis_train, loss_gen_train)
            sample_image = sample_gen.eval(feed_dict = {input_gen: sample(16, gen['input_dim']), drop_rate: 1})
            plot(sample_image)
    print('*** finish ***')