import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import matplotlib.pyplot as plt

# # create dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # read mnist data
train = {
    'data': mnist.train.images,
    'label': mnist.train.labels
}
test = {
    'data': mnist.test.images,
    'label': mnist.test.labels
}


# # config
epochs = 21
print_range = (epochs - 1) // 20
hiddenLayer1 = 512
hiddenLayer2 = 128
hiddenLayer3 = 10
inputDimension = train.get('data').shape[1] # 28*28 = 784
outputDimension = 2 # encode to 2 dimension for plot
batch_size = 256
batch_num = train.get('data').shape[0] // batch_size
start_learning_rate = 0.1
decay_steps = (epochs - 1) // 20
decay_rate = 0.9

# # create model
# ENCODE : 784 -> 512 -> 128 -> 10 -> 2
input_x = tf.placeholder(tf.float32, [None, inputDimension])
encode_W1 = tf.Variable(tf.zeros([inputDimension, hiddenLayer1]))
encode_b1 = tf.Variable(tf.zeros([hiddenLayer1]) + 0.01)
encode_y1 = tf.matmul(input_x, encode_W1) + encode_b1
encode_W2 = tf.Variable(tf.zeros([hiddenLayer1, hiddenLayer2]))
encode_b2 = tf.Variable(tf.zeros([hiddenLayer2]) + 0.01)
encode_y2 = tf.matmul(encode_y1, encode_W2) + encode_b2
encode_W3 = tf.Variable(tf.zeros([hiddenLayer2, hiddenLayer3]))
encode_b3 = tf.Variable(tf.zeros([hiddenLayer3]) + 0.01)
encode_y3 = tf.matmul(encode_y2, encode_W3) + encode_b3
encode_W4 = tf.Variable(tf.zeros([hiddenLayer3, outputDimension]))
encode_b4 = tf.Variable(tf.zeros([outputDimension]) + 0.01)
encode_result = tf.matmul(encode_y3, encode_W4) + encode_b4
# DECODE : 2 -> 10 -> 128 -> 512 -> 784
decode_W1 = tf.Variable(tf.zeros([outputDimension, hiddenLayer3]))
decode_b1 = tf.Variable(tf.zeros([hiddenLayer3]) + 0.01)
decode_y1 = tf.matmul(encode_result, decode_W1) + decode_b1
decode_W2 = tf.Variable(tf.zeros([hiddenLayer3, hiddenLayer2]))
decode_b2 = tf.Variable(tf.zeros([hiddenLayer2]) + 0.01)
decode_y2 = tf.matmul(decode_y1, decode_W2) + decode_b2
decode_W3 = tf.Variable(tf.zeros([hiddenLayer2, hiddenLayer1]))
decode_b3 = tf.Variable(tf.zeros([hiddenLayer1]) + 0.01)
decode_y3 = tf.matmul(decode_y2, decode_W3) + decode_b3
decode_W4 = tf.Variable(tf.zeros([hiddenLayer1, inputDimension]))
decode_b4 = tf.Variable(tf.zeros([inputDimension]) + 0.01)
decode_result = tf.matmul(decode_y3, decode_W4) + decode_b4
# loss: 
loss = tf.reduce_mean(tf.square(tf.subtract(input_x, decode_result)))
# optimizer: GradientDescent + decay learning rate
global_step = tf.placeholder(tf.int64)
learning_rate =  tf.train.exponential_decay(learning_rate=start_learning_rate, global_step=global_step, decay_steps=decay_steps, decay_rate=decay_rate)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# # Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(epochs):
    # min batch
    for i in range(batch_num):
        batch_data, batch_label = mnist.train.next_batch(batch_size)   
        sess.run(train_step, feed_dict={input_x:batch_data, global_step: epoch})

    # print train process
    if epoch%print_range == 0:
        loss_train = sess.run(loss, feed_dict={input_x:train.get('data'), global_step: epoch})
        loss_test = sess.run(loss, feed_dict={input_x:test.get('data'), global_step: epoch})
        print(datetime.now(), epoch)
        print('- Train: ', loss_train)
        print('-  Test: ', loss_test)

print('*** finish train ***')


# # plot training result
# Compare original images with decode result
examples_to_show = 5
decode_result = sess.run(decode_result, feed_dict={input_x: test.get('data')[:examples_to_show]})
f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(test.get('data')[i], (28, 28)))
    a[1][i].imshow(np.reshape(decode_result[i], (28, 28)))
plt.show()
# plot encode result for test data
encode_result = sess.run(encode_result, feed_dict={input_x: test.get('data')})
plt.scatter(encode_result[:, 0], encode_result[:, 1], c=test.get('labels'))
plt.colorbar()
plt.show()