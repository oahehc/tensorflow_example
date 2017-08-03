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

---

### autoencoder.py : reduce dimension for MNIST images cluster & visualization
- data  : MNIST
- layer : 28*28=784 -> 512 -> 256 -> 64 -> 2 -> 64 -> 256 -> 512 -> 784
- activation function : sigmoid
- optimizier : Adam + min-batch

---

### tsne_pca.py : reduce dimension for MNIST images cluster & visualization
- PCA
- t-SNE
- PCA + t-SNE : performance check






### CNN

- additional : L2 regularization
- optimizier : gradient descent + decayed learning rate + min-batch


### RNN



### CBOW
### Skip-Gram
