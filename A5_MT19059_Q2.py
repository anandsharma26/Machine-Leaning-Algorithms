#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=[
[1 ,0, -45],
[2 ,1 ,-51],
[3 ,2, -58],
[4 ,3, -63],
[5 ,4 ,-36],
[6 ,5 ,-52],
[7 ,6 ,-59],
[8 ,7 ,-62],
[9 ,8 ,-36],
[10 ,9, -43],
[11 ,10, -55],
[12 ,11 ,-64]
]
print((data))


# In[2]:


import math
def calculate_func(Xi,Xj,l,sigmaf):
    ans=math.exp(-0.5*((Xi-Xj)**2)/(l**2))
    ans=ans*(sigmaf)
    #print(Xi,Xj," ans ",ans)
    return ans


# In[3]:


def gaussian_process_regression(X,Y,l,sigmaf):
    X=np.array(X,dtype=float)
    Y=np.array(Y,dtype=float)
    n=len(X)

    K=[[0]*n]*n
    #print("Empty array",K)
    for i in range(n):
        for j in range(n):
            K[i][j]=(calculate_func(X[i],X[j],l,sigmaf))

    K=np.array(K)
    Kstar=[0]*n
    X_mean=sum(X)/len(X)
    for i in range(n):
        Kstar[i]=calculate_func(X[i],X_mean,l,sigmaf)
    Kstar=np.array(Kstar).reshape(n,1)
    
    Kstarstar=calculate_func(X_mean,X_mean,l,sigmaf)

    Kstar=Kstar.T
  
    print("K shape is",K.shape)
    #print(type(K))
    temp1=np.matmul(Kstar,np.linalg.pinv(K))
    mean=np.matmul(temp1,Y)
    #print("mean is ",mean)
    temp=np.matmul(temp1,Kstar.T)
    cov=Kstarstar-temp
    #print("Covariance is ",cov)
    return (mean),(cov)


# In[4]:


from numpy.linalg import cholesky,lstsq,det
from scipy.optimize import minimize
def optimize_main_func(X_train,Y_train):
    def optimize_theta(theta):
        #temp=0.5*Y_train.T.dot(np.linalg.pinv(K)).dot(Y_train)+0.5*len(X_train)*np.log(2*np.pi)
        n=len(X_train)
        K=[[0]*n]*n
        #print("Empty array",K)
        for i in range(n):
            for j in range(n):
                K[i][j]=(calculate_func(X_train[i],X_train[j],l=theta[0],sigmaf=theta[1]))

        K=np.array(K)
        
        L=cholesky(K)
        #print(L)
        return np.sum(np.log(np.diagonal(L))) +                    0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) +                    0.5 * len(X_train) * np.log(2*np.pi)
     
    return optimize_theta
    


# In[5]:


train_data=[]*7
test_data=[]*5

j=0
for row in data:
    if(row[0]%2==0 and row[0]!=12):
        test_data.append(row)
    else :
        train_data.append(row)
# print(test_data)
# print(train_data)
print(len(test_data))
print(test_data)

X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
for i in test_data:
    X_test.append(i[1])
    Y_test.append(i[2])
for i in train_data:
    X_train.append(i[1])
    Y_train.append(i[2])
train_data=np.array(train_data)
X_train=np.array(X_train)
Y_train=np.array(Y_train)
print("X_train",X_train)
print("Y_train", Y_train)

res = minimize(optimize_main_func(X_train, Y_train), [1, 1], 
               bounds=((1e-5, None), (1e-5, None)),method='Nelder-Mead')
l=1
sigmaf=1

l_opt, sigma_f_opt = res.x
print("l_opt",l_opt,"sigma_f_opt", sigma_f_opt)

X_train=X_train.tolist()
Y_train=Y_train.tolist()
Y_pred=[]
for i in X_test:
    mean,cov=gaussian_process_regression(X_train,Y_train,l_opt,sigma_f_opt)
    X_train.append(i)
    Y_train.append(mean)
    Y_pred.append(mean)
    

for i in range(len(X_test)):
    print(" X= ",X_test[i],"Y =",Y_pred[i])


# In[ ]:





# In[ ]:




