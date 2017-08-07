import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

# # DATASET
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # read mnist data
train = {
    'data': mnist.train.images,
    'label': mnist.train.labels
}
test = {
    'data': mnist.test.images,
    'label': mnist.test.labels
}
# print(train.get('data').shape)  # (55000, 784)
# print(test.get('data').shape)  # (10000, 784)


# # CONFIG
epochs = 101
print_range = (epochs - 1) // 20
hiddenLayer1 = 1024
hiddenLayer2 = 32
train_drop_rate = 0.5
inputDimension = train.get('data').shape[1] # 28*28 = 784
outputDimension = train.get('label').shape[1] # class 0~9 = 10
batch_num = 10
batch_size = train.get('data').shape[0] // batch_num


# # MODEL
drop_rate = tf.placeholder(tf.float32) # apply dropout to prevent overfitting
# first layer: input_x * W1 + b1 = y1, 784 -> 1024
input_x = tf.placeholder(tf.float32, [None, inputDimension])
W1 = tf.Variable(tf.zeros([inputDimension, hiddenLayer1]))
b1 = tf.Variable(tf.zeros([hiddenLayer1]) + 0.01)
y1 = tf.matmul(input_x, tf.nn.dropout(W1, drop_rate)) + b1
# second layer: y1 * W2 + b2 = y2, 1024 -> 32
W2 = tf.Variable(tf.zeros([hiddenLayer1, hiddenLayer2]))
b2 = tf.Variable(tf.zeros([hiddenLayer2]) + 0.01)
y2 = tf.matmul(y1, tf.nn.dropout(W2, drop_rate)) + b2
# output layer: softmax(y2 * W3 + b3) = prediction, 32 -> 10
W3 = tf.Variable(tf.zeros([hiddenLayer2, outputDimension]))
b3 = tf.Variable(tf.zeros([outputDimension]) + 0.01)
prediction = tf.nn.softmax(tf.matmul(y2, tf.nn.dropout(W3, drop_rate)) + b3)
# loss: use cross_entropy to calculate distance between two distribution
output_y = tf.placeholder(tf.float32, [None, outputDimension])
loss = tf.reduce_mean(-tf.reduce_sum(output_y*tf.log(prediction), axis=1))
# accuracy: choose max probability as prediction
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(output_y,1)), tf.float32))
# optimizer: Adam
train_step = tf.train.AdamOptimizer().minimize(loss)


# # TRAIN
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(epochs):
    # min batch
    for i in range(batch_num):
        batch_data, batch_label = mnist.train.next_batch(batch_size)   
        sess.run(train_step, feed_dict={input_x:batch_data, output_y:batch_label, drop_rate: train_drop_rate})

    # print train process
    if epoch%print_range == 0:
        accuracy_train = sess.run(accuracy, feed_dict={input_x:train.get('data'), output_y:train.get('label'), drop_rate: 1.0})
        accuracy_test = sess.run(accuracy, feed_dict={input_x:test.get('data'), output_y:test.get('label'), drop_rate: 1.0})
        print(datetime.now(), epoch)
        print('- Train: ', accuracy_train)
        print('-  Test: ', accuracy_test)

print('*** finish ***')