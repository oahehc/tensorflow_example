### multi_layer.py : classify number's images from 0 to 9
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
### autoencoder.py : reduce dimension for MNIST images cluster & visualization
- data  : MNIST
- layer : 28*28=784 -> 512 -> 256 -> 64 -> 2 -> 64 -> 256 -> 512 -> 784
- activation function : sigmoid
- optimizier : Adam + min-batch

```
epochs = 101
- Train Loss :  0.0713046
-  Test Loss :  0.0757816
```
- original image VS decoder
![Imgur](http://i.imgur.com/sleJQZK.png)

- cluster by encoder
![Imgur](http://i.imgur.com/KQih2JE.png)
---
### tsne_pca.py : reduce dimension for MNIST images cluster & visualization
- PCA
- t-SNE
- PCA + t-SNE : performance check






- additional : L2 regularization
- optimizier : gradient descent + decayed learning rate + min-batch


### CNN
### RNN
### CBOW
### Skip-Gram
### VAM
### GAN