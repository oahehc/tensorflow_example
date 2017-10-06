### activation_function.py
![Imgur](http://i.imgur.com/qh2ERFy.png)


--- 
### DNN.py - deep neural network
#### basic deep learning model to classify number's images from 0 to 9
- data  : MNIST
- layer : 28*28=784 -> 1024 -> 32 -> 10
- activation function : softmax
- additional : dropout
- optimizier : Adam + min-batch
```
epochs = 101
- Train Accuracy :  0.908382
-  Test Accuracy :  0.9068
```

---
### CNN.py - Convolutional Neuron Networks
#### apply CNN model to classify number's images from 0 to 9
- data  : MNIST
- layer : 28 * 28 * 1 -> 14 * 14 * 4 -> 7 * 7 * 8 -> 512 -> 10
- activation function : Relu, Sigmoid
- additional : dropout
- optimizier : Adam + min-batch
```
epochs = 11
      | Relu	  | Sigmoid   | None
Train | 0.990234    | 0.953125  | 0.984375
Test  | 0.9875	  | 0.9595    | 0.9826
*Train only measure min-batch
```

---
### TensorBoard.py
#### use CNN.py model to create tensorbaord for visualize model structure
- model structure<br>
![Imgur](http://i.imgur.com/7NPm9ls.png)
- layer detail<br>
![Imgur](http://i.imgur.com/fVC6MyP.png)

---
### Autoencoder.py
#### reduce dimension for MNIST images cluster & visualization
- data  : MNIST
- layer : 28*28=784 -> 512 -> 256 -> 64 -> 2 -> 64 -> 256 -> 512 -> 784
- activation function : Sigmoid
- optimizier : Adam + min-batch
```
epochs = 101
- Train Loss :  0.0713046
-  Test Loss :  0.0757816
```
- original image VS decoder<br>
![Imgur](http://i.imgur.com/sleJQZK.png)

- cluster by encoder<br>
![Imgur](http://i.imgur.com/KQih2JE.png)

---
### t-SNE_PCA.py - t-Distributed Stochastic Neighbor Embedding / Principal components analysis
#### reduce dimension for MNIST images cluster & visualization
- PCA
- t-SNE
- PCA + t-SNE : combine PCA and t-SNE to prevent performance issue when apply t-SNE with high dimension data
```
used_time
- PCA          0:00:00.782207
- t-SNE        0:04:43.643358
- PCA + t-SNE  0:03:29.427107
```
![Imgur](http://i.imgur.com/4yDlTsF.png)

---
### VAE.py - Variational Autoencoder
#### apply autoencoder to decode image, and generate new image by select new decode data
- data  : MNIST
- layer : 28*28=784 -> 512 -> 512 -> 2(mean, var) -> 512 -> 512 -> 784
- activation function : softplus, sigmoid
- optimizier : Adam + min-batch
```
Epoch: 1 cost= 196.581512416
Epoch: 21 cost= 146.854643957
Epoch: 41 cost= 142.512102212
Epoch: 61 cost= 140.136859013
Epoch: 81 cost= 138.670617768
Epoch: 101 cost= 137.499377324
```
- original image VS decoder<br>
![Imgur](http://i.imgur.com/qIyxc9L.png)
- generate images<br>
![Imgur](http://i.imgur.com/wkCjX2z.png)

---
### RNN.py - Recurrent Neural Networks
- data  : MNIST
- layer : encoder(time, data -> time, feature) > RNN(time, feature -> time, est.feature) > decoder(est.feature -> class)
- RNN type : 
  - basicRNN
  - LSTM (long short-term memory)
  - GRU (Gated Recurrent Units)

![Imgur](http://i.imgur.com/I4zAKOT.png)


---
### Skip-Gram.py
#### estimated probability of each words close to the target word
- data  : http://mattmahoney.net/dc/text8.zip
- layer : target word -> embedding to vector(embedding matrix * target word one-hot-encoding) 
            -> linear model(W*V + b) = probability for each word connect to target -> softmax -> loss = cross entropy with label
- additional : Sampled Softmax
- optimizier : AdagradOptimizer
```
*  20000 Average loss : 3.73424506655
*  40000 Average loss : 3.44429408045
*  60000 Average loss : 3.39420536309
*  80000 Average loss : 3.36515107585
* 100000 Average loss : 3.19453594169
Nearest to 'were'  : are, have, was, had, while, those, been, although,
Nearest to 'state' : city, states, territory, republic, shadowed, drum, gums, frites,
```
![Imgur](http://i.imgur.com/7QG7ktf.png)

---
### CBOW.py - Continuous Bag of Words
#### estimated target word base on input words
- data  : http://mattmahoney.net/dc/text8.zip
- layer : input words(connect to target) ->  sum(embedding input to vector) -> probability of target word -> softmax -> loss = cross entropy with label
- additional : Sampled Softmax
- optimizier : AdagradOptimizer
```
*  20000 Average loss : 3.12684643236
*  40000 Average loss : 2.92018117781
*  60000 Average loss : 2.83985439647
*  80000 Average loss : 2.76869805927
* 100000 Average loss : 2.57371024392
Nearest to eight : nine, seven, six, four, five, three, zero, two, 
Nearest to often : usually, sometimes, generally, commonly, frequently, typically, actually, now, 
```
![Imgur](http://i.imgur.com/mcQzxyH.png)

---
### GAN.py - Generative Adversarial Network
- data : MNIST
- generator : 128 -> 256 -> 784=28*28
- discriminator : 784 -> 256 -> 1
- activation function : softplus, sigmoid
- optimizier : Adam + min-batch
![Imgur](https://i.imgur.com/GY8KmzY.png)

---
### DCGAN.py - Deep Convolutional Generative Adversarial Network
- data : MNIST
- generator : 128 -> 7*7*64 -> 14*14*32 -> 28*28*1
- discriminator : 28*28*1 -> 14*14*32(strides=2) -> 7*7*64(strides=2) -> 1
- activation function : relu, tanh, sigmoid
- optimizier : Adam + min-batch
![Imgur](https://i.imgur.com/c2aREyw.png)

---
[TBD] ------------------------
### WGAN.py - Wasserstein Generative Adversarial Network

---
### Conditional GAN
---
### Transfer Learning
#### inception model
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/07_Inception_Model.ipynb
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/08_Transfer_Learning.ipynb


---
### Matrix Factorization

---
### Highway Network
### Attention-based Model
### Reinforcement Learning

---
- additional : L2 regularization
- additional : Xavier initialization