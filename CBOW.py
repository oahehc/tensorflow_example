'''
Reference
https://classroom.udacity.com/courses/ud730
https://github.com/rndbrtrnd/udacity-deep-learning/blob/master/5_word2vec.ipynb
'''

import tensorflow as tf
import numpy as np
from six.moves.urllib.request import urlretrieve  # download data by url
import zipfile  # read and write zipfile
import collections  # count and double-ended queue
%matplotlib inline


# # GET TEXT DATA
url = 'https://s3-ap-northeast-1.amazonaws.com/oahehc-dl/'
filename = 'text8.zip'
textfile, _ = urlretrieve(url + filename, filename)
with zipfile.ZipFile(textfile) as f:
    words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    print('Data size %d' % len(words))


# # CREATE VACABULARY DATASET
# count: [['UNK', 2735459], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)...]
# dictionary: {'UNK': 0, 'the': 1 ...}
# reverse_dictionary: {0: 'UNK', 1: 'the' ...}
# data: [0, 3084, 12, 6, 195, 2, 3137, 46, 59, 156, ...], transfer article to word index
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
del words # save memory
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])


# # GENERATE TRAINING BATCH
