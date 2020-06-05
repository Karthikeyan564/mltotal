import numpy as np
import random
class sn:
    def __init__(self,sizes):
        self.lay = len(sizes)
        self.sizes=sizes
        self.c=0
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[1:],sizes[:-1])]
    def costd(self,out,y):
        """Derivative of the cost function"""
        return out-y
    def bp(self,x,y):
        """Return the derivatives of C wrt weights and biases"""
        bd = [np.zeros(b.shape) for b in self.biases]
        wd = [np.zeros(w.shape) for w in self.biases]
        #forward propagation
        a = x
        an = [x]
        zn = []

        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,a)+b
            a = sig(z)
            zn.append(z)
            an.append(a)
        #backward propagation
        delta = self.costd(an[-1],y)*sigd(zn[-1])
        bd[-1] = delta
        wd[-1] = np.dot(delta,an[-2].transpose())
        for l in range(2,self.lay):
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sigd(zn[-l])
            bd[-l] = delta
            wd[-l] = np.dot(delta,an[-l-1].transpose())
        return wd,bd
    def batche(self,bat,rate):
        "mini-batch training"
        bd = [np.zeros(b.shape) for b in self.biases]
        wd = [np.zeros(w.shape) for w in self.weights]
        for x,y in bat:
            wb,bb = self.bp(x,y)
            bd = [ol+ne for ol,ne in zip(bd,bb)]
            wd = [ol+ne for ol,ne in zip(wd,wb)]
        self.weights = [w-(rate/len(bat))*i for w,i in zip(self.weights,wd)]
        self.biases = [b-(rate/len(bat))*i for b,i in zip(self.biases,bd)]
    def sgd(self,train,epoch,size,rate,test=None):
        for i in range(epoch):
            random.shuffle(train)
            batches = [train[k:k+size] for k in range(0,len(train),size)]
            for j in batches:
                self.batche(j,rate)
            print("Epoch",i+1,":",self.evalt(test),":",len(test))
    def fp(self,a):
        for w,b in zip(self.weights,self.biases):
            a = sig(np.dot(w,a)+b)
        return a
    def evalt(self,test):
        pred = [(np.argmax(self.fp(x)),np.argmax(y)) for x,y in test]
        return sum(int(x==y) for x,y in pred)
    def wp(self):
        return self.weights,self.biases
def sig(z):
    """The sigmoid function"""
    return 1.0/(1.0+np.exp(-z))
def sigd(z):
    """Derivative of the sigmoid function"""
    return sig(z)*(1-sig(z))