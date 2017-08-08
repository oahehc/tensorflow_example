'''
Reference: 
https://www.youtube.com/watch?v=8zomhgKrsmQ&t=112s
http://blog.csdn.net/jackytintin/article/details/53641885
https://jmetzen.github.io/2015-11-27/vae.html
https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/vae/vae_demo.ipynb
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline

def weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias(shape):
    return tf.Variable(tf.zeros(shape) + 0.1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# # CONFIG
save_filename = 'model/model.ckpt'
n_samples = mnist.train.num_examples
n_z = 2 # Dimension of the latent space
layer_1 = 512
layer_2 = 512
batch_size = 128
total_batch = n_samples // batch_size
epochs = 101
print_range = (epochs - 1) // 10


# # ENCODER
x = tf.placeholder(tf.float32, shape=[None, 28*28])
W_fc1 = weights([28*28, layer_1])
b_fc1 = bias([layer_1])
h_1   = tf.nn.softplus(tf.matmul(x, W_fc1) + b_fc1)
W_fc2 = weights([layer_1, layer_2])
b_fc2 = bias([layer_2])
h_2   = tf.nn.softplus(tf.matmul(h_1, W_fc2) + b_fc2)
W_mean = weights([layer_2, n_z])
b_mean = bias([n_z])
W_sigma = weights([layer_2, n_z])
b_sigma = bias([n_z])
z_mean = tf.add(tf.matmul(h_2, W_mean), b_mean)
z_log_sigma_sq = tf.add(tf.matmul(h_2, W_sigma), b_sigma)
input_data_num = tf.placeholder(tf.int64)
eps = tf.random_normal((input_data_num, n_z), 0, 1, dtype=tf.float32) # random number generate from N(0,1)
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # mean + noise
# # DECODER
W_fc1_g = weights([n_z, layer_2])
b_fc1_g = bias([layer_2])
h_1_g   = tf.nn.softplus(tf.matmul(z, W_fc1_g) + b_fc1_g)
W_fc2_g = weights([layer_2, layer_1])
b_fc2_g = bias([layer_1])
h_2_g   = tf.nn.softplus(tf.matmul(h_1_g, W_fc2_g) + b_fc2_g)
x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(h_2_g, weights([layer_1, 28*28])), bias([28*28])))
# # LOSS
cross_entropy = -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean) + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean), axis=1)
kl_divergence = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), axis=1) # maximize likelihood base on current x => minimize KL-divergence
cost = tf.reduce_mean(cross_entropy + kl_divergence)
optimizer =  tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


# # TRAIN
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
for epoch in range(epochs):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run((optimizer, cost), feed_dict={x: batch_xs, input_data_num: batch_size})
        total_cost += cost_val
    if epoch%print_range == 0:
        print('Epoch:', epoch+1, 'cost=', total_cost/total_batch)

save_path = saver.save(sess, save_filename)
print('Model saved in file: ', format(save_path))


# # PLOT : mapping original image with decode result
saver = tf.train.Saver()
check_num = 5
plot_data_num = 2000
with tf.Session() as sess:
    saver.restore(sess, save_filename)
    print('Model restored.')
    x_sample, x_label = mnist.test.next_batch(plot_data_num)
    x_reconstruct, z_vals, z_mean_val = sess.run((x_reconstr_mean, z, z_mean), feed_dict={x: x_sample, input_data_num: plot_data_num})

    plt.figure(figsize=(8, check_num*2))
    for i in range(check_num):
        plt.subplot(check_num, 3, 3*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1,  interpolation='none', cmap=plt.get_cmap('gray'))
        plt.title('Original')
        
        plt.subplot(check_num, 3, 3*i + 2)
        plt.scatter(z_vals[:,0],z_vals[:,1], c=np.argmax(x_label, axis=1), s=10, alpha=.2) # mnist data cluster
        plt.colorbar()
        plt.scatter(z_mean_val[i,0], z_mean_val[i,1], c='black', s=30, alpha=.9) # mean
        plt.scatter(z_vals[i,0],z_vals[i,1], c='red', s=30, alpha=.9) # mean + noise
        plt.xlim((-3,3))
        plt.ylim((-3,3))
        plt.title('Latent Space')
        
        plt.subplot(check_num, 3, 3*i + 3)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, interpolation='none', cmap=plt.get_cmap('gray'))
        plt.title('Decode')
    plt.tight_layout()


# # PLOT : generate images
num = 20
linspace = np.linspace(-3, 3, num)
canvas = np.empty((28*num, 28*num))
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, save_filename)
    d = np.zeros([batch_size,2],dtype='float32')
    for i, yi in enumerate(linspace):
        for j, xi in enumerate(linspace):
            z_mu = np.array([[xi, yi]])
            d[0] = z_mu
            x_mean = sess.run(x_reconstr_mean, feed_dict={z: d})
            canvas[(num-i-1)*28:(num-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))        
Xi, Yi = np.meshgrid(linspace, linspace)
plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
plt.tight_layout()