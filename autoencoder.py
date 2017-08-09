import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import matplotlib.pyplot as plt

# # DATASET
mnist = input_data.read_data_sets(
    'MNIST_data', one_hot=True)  # read mnist data
train = {
    'data': mnist.train.images,
    'label': mnist.train.labels
}
test = {
    'data': mnist.test.images,
    'label': mnist.test.labels
}


# # CONFIG
epochs = 101
print_range = (epochs - 1) // 10
hiddenLayer1 = 512
hiddenLayer2 = 256
hiddenLayer3 = 64
inputDimension = train.get('data').shape[1]  # 28*28 = 784
outputDimension = 2  # encode to 2 dimension for plot result
batch_size = 256
batch_num = train.get('data').shape[0] // batch_size


# # MODEL
# ENCODE : 784 -> 512 -> 256 -> 64 -> 2
input_x = tf.placeholder(tf.float32, [None, inputDimension])
encode_W1 = tf.Variable(tf.truncated_normal([inputDimension, hiddenLayer1]))
encode_b1 = tf.Variable(tf.truncated_normal([hiddenLayer1]))
encode_y1 = tf.nn.sigmoid(tf.matmul(input_x, encode_W1) + encode_b1)
encode_W2 = tf.Variable(tf.truncated_normal([hiddenLayer1, hiddenLayer2]))
encode_b2 = tf.Variable(tf.truncated_normal([hiddenLayer2]))
encode_y2 = tf.nn.sigmoid(tf.matmul(encode_y1, encode_W2) + encode_b2)
encode_W3 = tf.Variable(tf.truncated_normal([hiddenLayer2, hiddenLayer3]))
encode_b3 = tf.Variable(tf.truncated_normal([hiddenLayer3]))
encode_y3 = tf.nn.sigmoid(tf.matmul(encode_y2, encode_W3) + encode_b3)
encode_W4 = tf.Variable(tf.truncated_normal([hiddenLayer3, outputDimension]))
encode_b4 = tf.Variable(tf.truncated_normal([outputDimension]))
encode_result = tf.matmul(encode_y3, encode_W4) + encode_b4
# DECODE : 2 -> 64 -> 256 -> 512 -> 784
decode_W1 = tf.Variable(tf.truncated_normal([outputDimension, hiddenLayer3]))
decode_b1 = tf.Variable(tf.truncated_normal([hiddenLayer3]))
decode_y1 = tf.nn.sigmoid(tf.matmul(encode_result, decode_W1) + decode_b1)
decode_W2 = tf.Variable(tf.truncated_normal([hiddenLayer3, hiddenLayer2]))
decode_b2 = tf.Variable(tf.truncated_normal([hiddenLayer2]))
decode_y2 = tf.nn.sigmoid(tf.matmul(decode_y1, decode_W2) + decode_b2)
decode_W3 = tf.Variable(tf.truncated_normal([hiddenLayer2, hiddenLayer1]))
decode_b3 = tf.Variable(tf.truncated_normal([hiddenLayer1]))
decode_y3 = tf.nn.sigmoid(tf.matmul(decode_y2, decode_W3) + decode_b3)
decode_W4 = tf.Variable(tf.truncated_normal([hiddenLayer1, inputDimension]))
decode_b4 = tf.Variable(tf.truncated_normal([inputDimension]))
decode_result = tf.nn.sigmoid(tf.matmul(decode_y3, decode_W4) + decode_b4)
# loss:
loss = tf.reduce_mean(tf.square(input_x - decode_result))
# optimizer: 
train_step = tf.train.AdamOptimizer().minimize(loss)


# # TRAIN
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        # min batch
        for i in range(batch_num):
            batch_data, batch_label = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={input_x: batch_data})

        # print train process
        if epoch % print_range == 0:
            loss_train = sess.run(loss, feed_dict={input_x: train.get('data')})
            loss_test = sess.run(loss, feed_dict={input_x: test.get('data')})
            print(datetime.now(), epoch)
            print('- Train: ', loss_train)
            print('-  Test: ', loss_test)

    test_decode_result = sess.run(decode_result, feed_dict={
                                input_x: test.get('data')})
    test_encode_result = sess.run(encode_result, feed_dict={
                                input_x: mnist.test.images})
    print('*** finish train ***')


# # PLOT
# plot encode result for test data
plt.scatter(test_encode_result[:, 0], test_encode_result[:, 1], c=np.argmax(
    test.get('labels'), axis=1))
plt.colorbar()
plt.show()
# Compare original images with decode result
examples_num = 5
f, a = plt.subplots(2, examples_num, figsize=(examples_num, 2))
for i in range(examples_num):
    a[0][i].imshow(np.reshape(test.get('data')[i], (28, 28)))
    a[1][i].imshow(np.reshape(test_decode_result[i], (28, 28)))
plt.show()
