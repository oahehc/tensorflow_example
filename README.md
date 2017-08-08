### activation_function
![Imgur](http://i.imgur.com/qh2ERFy.png)


--- 
### Basic_Model.py
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
### CNN.py
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
### t-SNE_PCA.py
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
### VAE.py
#### 
- data  : MNIST
- layer : 28*28=784 -> 512 -> 512 -> 2(mean, var) -> 512 -> 512 -> 784
- activation function : softplus, sigmoid
- optimizier : Adam + min-batch
```
Epoch: 1 cost= 197.138191134
Epoch: 11 cost= 153.145961379
Epoch: 21 cost= 147.470246073
Epoch: 31 cost= 144.192750588
Epoch: 41 cost= 142.880368568
Epoch: 51 cost= 141.337800495
Epoch: 61 cost= 140.225197319
Epoch: 71 cost= 139.715160308
Epoch: 81 cost= 139.138251511
Epoch: 91 cost= 138.459720549
Epoch: 101 cost= 137.795099627
```
- original image VS decoder<br>
![Imgur](http://i.imgur.com/tZLb76r.png)
- generate images<br>
![Imgur](http://i.imgur.com/pctUmsS.png)


---

- additional : L2 regularization
- additional : Xavier initialization
- optimizier : decayed learning rate



### VAE
### GAN
### RNN
### CBOW
### Skip-Gram