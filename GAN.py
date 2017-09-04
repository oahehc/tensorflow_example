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

# # DATASET
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # read mnist data


# # DEFINE FUNCTION
def weight_variable(shape, name='var'):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name)
def bias_variable(shape, name='var'):
    return tf.Variable(tf.zeros(shape) + 0.1, name)


# # CONFIG
epochs = 1001
print_range = (epochs - 1) // 10
batch_size = 256
batch_num = mnist.train.images.shape[0] // batch_size
train_drop_rate = 0.5
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
input_gen = tf.placeholder(tf.float32, [None, gen['input_dim']])
W1_gen = weight_variable([gen['input_dim'], gen['layer1_dim']])
b1_gen = bias_variable([gen['layer1_dim']])
W2_gen = weight_variable([gen['layer1_dim'], gen['output_dim']])
b2_gen = bias_variable([gen['output_dim']])
var_gen = [W1_gen, b1_gen, W2_gen, b2_gen]
def generator(inputData):
    h1_gen = tf.nn.softplus(tf.matmul(inputData, W1_gen) + b1_gen)
    output_gen = tf.nn.sigmoid(tf.matmul(h1_gen, W2_gen) + b2_gen)
    return output_gen

input_dis = tf.placeholder(tf.float32, [None, dis['input_dim']])
W1_dis = weight_variable([dis['input_dim'], dis['layer1_dim']])
b1_dis = bias_variable([dis['layer1_dim']])
W2_dis = weight_variable([dis['layer1_dim'], dis['output_dim']])
b2_dis = bias_variable([dis['output_dim']])
var_dis = [W1_dis, b1_dis, W2_dis, b2_dis]
def discriminator(inputData):
    h1_dis = tf.nn.softplus(tf.matmul(inputData, W1_dis) + b1_dis)
    output_dis = tf.nn.sigmoid(tf.matmul(h1_dis, W2_dis) + b2_dis)
    return output_dis

def sample(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

sample_gen = generator(input_gen)
dis_real = discriminator(input_dis)
dis_fake = discriminator(sample_gen)
# loss
loss_dis = -tf.reduce_mean(tf.log(dis_real) + tf.log(1. - dis_fake))
loss_gen = -tf.reduce_mean(tf.log(dis_fake))
optimizer_dis = tf.train.AdamOptimizer(0.0001).minimize(loss_dis, var_list= var_dis)
optimizer_gen = tf.train.AdamOptimizer(0.0001).minimize(loss_gen, var_list= var_gen)


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


# # TRAIN
min_loss = 0.001 # early stop if loss goes to 0
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        for i in range(batch_num):
            batch_x = mnist.train.next_batch(batch_size)[0]
            _, loss_dis_train = sess.run([optimizer_dis, loss_dis], feed_dict = {input_dis: batch_x, input_gen: sample(batch_size, gen['input_dim'])})
            _, loss_gen_train = sess.run([optimizer_gen, loss_gen], feed_dict = {input_gen: sample(batch_size, gen['input_dim'])})
        if loss_dis_train < min_loss:
            print('early stop ', epoch, loss_dis_train, loss_gen_train)
            break
        elif epoch%print_range == 0:
            print(datetime.now(), epoch, loss_dis_train, loss_gen_train)
    sample_image = sample_gen.eval(feed_dict = {input_gen: sample(16, gen['input_dim'])})
    plot(sample_image)
    print('*** finish ***')





# # # CONFIG
# epochs = 11
# print_range = (epochs - 1) // 10
# inputDimension = mnist.train.images.shape[1] # 28*28 = 784
# outputDimension = 1 # real image percentage
# batch_size = 64
# batch_num = mnist.train.images.shape[0] // batch_size
# layer1_FilterSize = 5
# layer1_Depth = 4
# layer2_FilterSize = 5
# layer2_Depth = 8
# activation = tf.nn.relu
# train_drop_rate = 0.5
# inputDim_gen = 128


# # # MODEL: generator
# # 128 > 728
# input_gen = tf.placeholder(tf.float32, [None, inputDim_gen])
# W1_gen = weight_variable([inputDim_gen, inputDimension])
# b1_gen = bias_variable([inputDimension])    
# output_gen = tf.nn.sigmoid(tf.matmul(input_gen, W1_gen) + b1_gen)


# # # MODEL: discriminator
# # 728 > 28*28*1 > 14*14*4 > 7*7*8 > 1
# input_x = tf.placeholder(tf.float32, [None, inputDimension])
# input_x_2d = tf.reshape(input_x, [-1, 28, 28, 1]) # reshape to 2-dimension
# drop_rate = tf.placeholder(tf.float32)
# # layer1 conv : sample padding 5*5 2d-convolution + 2*2 maxpool
# W1_conv = weight_variable([layer1_FilterSize, layer1_FilterSize, 1, layer1_Depth])
# b1_conv = bias_variable([layer1_Depth])
# h1_conv = activationFunc(tf.nn.conv2d(input_x_2d, W1_conv, strides=[1, 1, 1, 1], padding='SAME') + b1_conv, activation)
# y1_conv = tf.nn.max_pool(h1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# # layer2 conv : sample padding 5*5 2d-convolution + 2*2 maxpool
# W2_conv = weight_variable([layer2_FilterSize, layer2_FilterSize, layer1_Depth, layer2_Depth])
# b2_conv = bias_variable([layer2_Depth])
# h2_conv = activationFunc(tf.nn.conv2d(y1_conv, W2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv, activation)
# y2_conv = tf.nn.max_pool(h2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# y2_conv_flat = tf.reshape(y2_conv, [-1, 7*7*layer2_Depth])
# # layer3 full connect output layer : 
# W3_fc = weight_variable([7*7*layer2_Depth, outputDimension])
# b3_fc = bias_variable([outputDimension])                                          
# prediction = tf.nn.sigmoid(tf.matmul(y2_conv_flat, tf.nn.dropout(W3_fc, drop_rate)) + b3_fc)
# var_dis = [W1_conv, b1_conv, W2_conv, b2_conv, W3_fc, b3_fc]
# # loss: use cross_entropy to calculate distance between two distribution
# output_y = tf.placeholder(tf.float32, [None, outputDimension])
# loss = tf.reduce_mean(-tf.reduce_sum(output_y*tf.log(prediction), axis=1))
# train_step = tf.train.AdamOptimizer().minimize(loss, var_list = var_dis)


# # # TRAIN
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     for epoch in range(epochs):
#         # min batch
#         for i in range(batch_num):
#             batch_data, batch_label = mnist.train.next_batch(batch_size)   
#             sess.run(train_step, feed_dict={input_x:batch_data, output_y:batch_label, drop_rate: train_drop_rate})

#         # print train process
#         if epoch%print_range == 0:
#             print(datetime.now(), epoch)
#     print('*** finish ***')


#     # # TRAINING RESULT TEST
#     test_num = 9
#     test_dataset = mnist.test.images
#     test_image = test_dataset[0:test_num, ]
#     test_result = sess.run(tf.argmax(prediction,1), feed_dict={input_x:test_image, drop_rate: 1.0})
#     fig, axes = plt.subplots(3, 3)
#     for i, ax in enumerate(axes.flat):
#         ax.imshow(np.reshape(test_image[i], (28, 28)))
#         xlabel = "Pred: {0}".format(test_result[i])
#         ax.set_xlabel(xlabel)
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.show()