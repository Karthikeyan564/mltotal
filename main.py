import numpy as np
import idx2numpy as dx
import matplotlib.pyplot as plt
import random
import nn
x1 = dx.convert_from_file("data/train-images.idx3-ubyte")
y1 = dx.convert_from_file("data/train-labels.idx1-ubyte")
xt = dx.convert_from_file("data/t10k-images.idx3-ubyte")
yt = dx.convert_from_file("data/t10k-labels.idx1-ubyte")
plt.imshow(x1[5],cmap="gray")
plt.show()
print(y1[5])
plt.imshow(xt[5],cmap="gray")
plt.show()
print(yt[5])
x1=np.array([ i.ravel() for i in x1]).reshape(60000,784,1)
xt=np.array([ i.ravel() for i in xt]).reshape(10000,784,1)
y1=np.array([(np.array(range(10))==y).astype("float").reshape(10,1) for y in y1])
yt=np.array([(np.array(range(10))==y).astype("float").reshape(10,1) for y in yt])
x1=x1/256
xt=xt/256
net= nn.sn([784,30,10])
net.sgd(list(zip(x1,y1)),10,10,3.0,list(zip(xt,yt)))
re=xt.reshape(10000,28,28)
plt.imshow(re[6],cmap="gray")
plt.show()
print(np.argmax(net.fp(xt[6])))