### multi_layer.py : classify number's images from 0 to 9
- data  : MNIST
- layer : 28*28=784 -> 1024 -> 32 -> 10
- activation function : softmax
- additional : dropout
- optimizier : Adam + min-batch
```
epochs = 101
- Train:  0.908382
-  Test:  0.9068
```

### autoencoder.py : reduce dimension for MNIST images cluster
- data  : MNIST
- layer : 28*28=784 -> 512 -> 10 -> 2 -> 10 -> 512 -> 784
- activation function : sigmoid
- optimizier : gradient descent + decayed learning rate + min-batch


### tsne_pca.py : reduce dimension for MNIST images cluster
- PCA
- t-SNE
- PCA + t-SNE : performance check


### CNN

- additional : L2 regularization



### RNN



### CBOW
### Skip-Gram
