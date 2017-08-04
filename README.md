### multi_layer.py
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
### cnn.py
#### apply CNN model to classify number's images from 0 to 9
- data  : MNIST
- layer : 28 * 28 * 1 -> 14 * 14 * 4 -> 7 * 7 * 8 -> 512 -> 10
- activation function : Relu, Sigmoid
- additional : dropout
- optimizier : Adam + min-batch
```
epochs = 11
*Train only measure min-batch
      | Relu	  | Sigmoid   | None
Train | 0.990234  | 0.953125  | 0.984375
Test  | 0.9875	  | 0.9595    | 0.9826
```

--- 
### autoencoder.py
#### reduce dimension for MNIST images cluster & visualization
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
### tsne_pca.py
#### reduce dimension for MNIST images cluster & visualization
- PCA
- t-SNE
- PCA + t-SNE : combine PCA and t-SNE to prevent performance issue when apply t-SNE with high dimension data
```
used_time AND result
- PCA          0:00:00.782207
- t-SNE        0:04:43.643358
- PCA + t-SNE  0:03:29.427107
```
![Imgur](http://i.imgur.com/4yDlTsF.png)

---






- additional : L2 regularization
- optimizier : gradient descent + decayed learning rate + min-batch



### RNN
### CBOW
### Skip-Gram
### VAM
### GAN