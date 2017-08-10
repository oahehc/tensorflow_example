'''
Reference
https://classroom.udacity.com/courses/ud730
https://github.com/rndbrtrnd/udacity-deep-learning/blob/master/5_word2vec.ipynb
https://www.tensorflow.org/extras/candidate_sampling.pdf
'''
import tensorflow as tf
import numpy as np
from six.moves.urllib.request import urlretrieve  # download data by url
import zipfile  # read and write zipfile
import collections  # count and double-ended queue
import random
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline


# # GET TEXT DATA
url = 'https://s3-ap-northeast-1.amazonaws.com/oahehc-dl/'
filename = 'text8.zip'
textfile, _ = urlretrieve(url + filename, filename)
with zipfile.ZipFile(textfile) as f:
    words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    print('Data size', len(words))


# # CREATE VACABULARY DATASET
# count: [['UNK', 2735459], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)...]
# dictionary: {'UNK': 0, 'the': 1 ...}
# reverse_dictionary: {0: 'UNK', 1: 'the' ...}
# data: [0, 3084, 12, 6, 195, 2, 3137, 46, 59, 156, ...], transfer article
# to word index
vocabulary_size = 50000
count = [['UNK', -1]]
# count and get most common 50000-1 words
count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
# create dictionary for summary word and ranking
dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)
# reverse by use ranking as key
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
# translate words data to ranking number
data = list()
unk_count = 0
for word in words:
    if word in dictionary:
        index = dictionary[word]
    else:
        index = 0  # dictionary['UNK']
        unk_count = unk_count + 1
    data.append(index)
count[0][1] = unk_count  # count UNK number
del words  # save memory
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])


# # GENERATE TRAINING BATCH
data_index = 0


def generate_batch(batch_size, scan_num, scan_window):
    '''
    input
        batch_size: how many words need to be selected
        scan_num: how many words need to collect for each target -> batch_size / scan_num = how many target words in each batch
        scan_window: limit collect range = target word +- scan_window
    output
        batch: target word 
        labels: words relate to target
    example 
        *translate from index to word for easy understanding
        data : ['UNK', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', ...]
        batch_size = 8, scan_num = 4, scan_window = 2
            batch: ['as', 'as', 'as', 'as', 'a', 'a', 'a', 'a']
            labels: ['originated', 'a', 'term', 'UNK', 'term', 'as', 'of', 'originated']
    '''
    global data_index
    assert batch_size % scan_num == 0
    assert scan_num <= 2 * scan_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * scan_window + 1  # [ scan_window, target, scan_window ]
    # collect data within limit range (target word +- scan_window) as buffer
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # collect by word
    # loop for word, batch_size = number of data need to be select, scan_num =
    # number of word should select for each target
    for i in range(batch_size // scan_num):
        select = scan_window  # scan_window is center for buffer, will point to target word
        # create list to avoid select target or duplicate word
        selected_list = [scan_window]
        for j in range(scan_num):  # collect related word with target, collect amount = scan_num
            while select in selected_list:  # random select word within span range
                select = random.randint(0, span - 1)
            selected_list.append(select)  # add selected word to avoid list
            # add target word to batch
            batch[i * scan_num + j] = buffer[scan_window]
            # add related word to labels
            labels[i * scan_num + j, 0] = buffer[select]
        # append next data to buffer, first element will auto remove base on
        # maxlen setting
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)  # move to next word
    return batch, labels


# # CONFIG
word_vector_dim = 128 # Dimension of the word vector
batch_size = 128
scan_window = 1
scan_num = 2
num_sampled = 64 # The number of classes to randomly sample per batch
num_steps = 100001
print_range = (num_steps-1) // 10
# random select target to calculate nearest word base on current train result
valid_size = 8
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size)) # random select 16 index from 0~99 for validation


# # MODEL
# Input data.
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])  # 128*1
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])  # 128*1*1
valid_dataset = tf.constant(valid_examples, dtype=tf.int32) # 16*1 random list from 0~100

# Variables.
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, word_vector_dim], -1.0, 1.0)) # 50,000*128 = vector for all words
embed = tf.nn.embedding_lookup(embeddings, train_dataset) # transfer train_dataset to word vector (128*1) => (128*128)
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, word_vector_dim], stddev=1.0 / math.sqrt(word_vector_dim))) # 50,000*128
softmax_biases = tf.Variable(tf.zeros([vocabulary_size])) # 50,000*1

# weights(50,000*128) X inputs(128*128) + biases(50,000*1) => 50,000*128 = result for each trainning sample
    # weights: [num_classes, word_vector_dim]    (50,000*128)
    # biases: [num_classes]                      (50,000*1)
    # inputs: [batch_size, word_vector_dim]      (128*128)
    # labels: [batch_size, num_true]             (128*1)
    # num_sampled: An int. The number of classes to randomly sample per batch. (64)
    # num_classes: An int. The number of possible classes.                     (50,000)
    # num_true: An int. The number of target classes per training example.     (1)
    # => output[0] = 50,000*1 = the propability for each word near to target 
    #    labels[0] = 1*1 = index of correct answer
# apply sampled_softmax_loss for faster train
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

# random select target words and check nearest word base on current training result
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) # calculate vector length
normalized_embeddings = embeddings / norm # normalize embeddings
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))  # consine = V1*V2 / |V1|*|V2|


# # TRAIN
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, scan_num, scan_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % print_range == 0 and step > 0:
            # calculate loss
            average_loss = average_loss / print_range
            print('*', step, 'Average loss :', average_loss)
            average_loss = 0
            # print most close word base on consine between two vector
            sim = sess.run(similarity, feed_dict=feed_dict)
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1] # argsort : sort array and get original index, bypass 0(self)
                close_words = ''
                for k in range(top_k):
                    close_words += reverse_dictionary[nearest[k]] + ', '
                print('Nearest to', valid_word, ':', close_words)
    final_embeddings = normalized_embeddings.eval() # word vector for all words


# PLOT by t-SNE
num_points = 400  # plot first 400 words
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# embed result to 2-dimention point for plot
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :]) # bypass first word 'UNK'
def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                       textcoords='offset points', ha='right', va='bottom')
    plt.show()

words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)
