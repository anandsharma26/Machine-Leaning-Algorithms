#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # New Section

# In[244]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.setrecursionlimit(2500)


# In[246]:


name=input("Enter datset name")
#path=pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv")
path=pd.read_csv(name)
data=pd.DataFrame(path)
data['pm2.5'].fillna(data['pm2.5'].mode()[0],inplace=True)
lb=LabelEncoder()
data['cbwd']=lb.fit_transform(data['cbwd'])
data.drop(['No'],axis=1,inplace=True)
#pd.to_numeric(data['cbwd'])
data=data.sample(frac=1.0)
# print(data.shape)
# print(data.head())

X1=data.values.tolist()
#print(len(X1))
X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
train=[]
test=[]
for row in X1:
    if(row[0]==2010 or row[0]==2012):
        X_train.append(row)
        Y_train.append(row[4])
        train.append(row)
    elif(row[0]==2011 or row[0]==2013):
        X_test.append(row)
        Y_test.append(row[4])
        test.append(row)

X_train=np.array(X_train)
X_train=np.delete(X_train,4,1)
X_train=(X_train[:,:]).tolist()

X_test=np.array(X_test)
X_test=np.delete(X_test,4,1)
X_test=(X_test[:,:]).tolist()

# print(len(X_train))
# print(len(X_test))
# print(len(Y_train))
# print(len(Y_test))
# print(len(train))
# print(len(test))
#print(X_train)


# In[228]:


def group_split(index,value,dataset):
    left=list()
    right=list()
    for row in ((dataset)):
        if row[index]>value:
            right.append(row)
        else:
            left.append(row)
    return left,right


# In[229]:


def calculate_variance(groups):
    left,right=groups[0],groups[1]
    left_y=[i[4] for i in left]
    right_y=[i[4] for i in right]
    mean_left=np.mean(left_y)
    mean_right=np.mean(right_y)
    #print(left_y)
    
#     variance_left=np.var(left_y)
#     variance_right=np.var(right_y)
    variance_left=0.0
    variance_right=0.0
    for i in left_y:
        variance_left+=(i-mean_left)**2
    for i in right_y:
        variance_right+=(i-mean_right)**2
    #print(variance_left)
    #print(variance_right)
    total_len=len(left)+len(right)
    #gini=variance_left+variance_right
    gini=((len(left))*(variance_left)+(len(right))*(variance_right))/total_len
    return gini


# In[230]:


def get_best_split(X):
    b_index,b_value,b_score,b_groups=None,None,100000000,None
    for j in range(len(X[0])):
        if j==4:
            continue
        feature_list=[row[j] for row in X]
        unique_vals=set(feature_list)
        for value in unique_vals:
            groups=group_split(j,value,X)
            variance=calculate_variance(groups)
            #print("i th row and j th column",gini)
            if(variance<b_score):
                b_score=variance
                b_groups=groups
                b_index=j
                b_value=value
    return {'index':b_index,'value':b_value,'groups':b_groups}


# In[231]:


def terminal_node(root):
    #print(root)
    outcomes = [row[4] for row in root]
    #print(outcomes)
    return np.mean(outcomes)
    #return max(set(outcomes), key=outcomes.count) if outcomes else None


# In[232]:


def split_func(node,maxdepth,min_size,depth):
    left,right=node['groups']
    del(node['groups'])
    if not left or not right:
        node['left']=terminal_node(left+right)
        node['right']=terminal_node(left+right)
        return
    if depth>=maxdepth:
        node['left']=terminal_node(left)
        node['right']=terminal_node(right)
        return 
    if len(left)<=min_size:
        node['left']=terminal_node(left)
    else:
        node['left']=get_best_split(left)
        split_func(node['left'],maxdepth,min_size,depth+1)
    
    if len(right)<=min_size:
        node['right']=terminal_node(right)
    else:
        node['right']=get_best_split(right)
        split_func(node['right'],maxdepth,min_size,depth+1)


# In[233]:


def decision_tree_Build(train,maxdepth,min_size):
    root=get_best_split(train)
    split_func(root,maxdepth,min_size,1)
    #print("done")
    return root


# In[234]:


def predict(node ,row):
    if row[node['index']]<node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            return node['right']
            


# In[235]:


def accuracy_metric_func(actual,predicted):

    n=len(actual)
    sum_temp=0.0
    for i in range(n):
        sum_temp+=((actual[i]-predicted[i])**2)
        #print(sum_temp)
        
    return float(sum_temp/n) 
    
#     n=len(actual)
#     count=0
#     for i in range(n):
#         if predicted[i]-3<=actual[i]<=predicted[i]+3:
#             count+=1
#     return float(count/n)*100


# In[236]:


def test_func(tree,test):
    predictions_list=[]
    actual=[]
    for row in test:
        actual.append(row[4])
        average_pred=predict(tree,row)
        predictions_list.append(average_pred)

    score=accuracy_metric_func(actual,predictions_list)
    return predictions_list,score


# In[247]:


def Decision_Tree_regressor(train,test,max_depth,min_size):
    tree=decision_tree_Build(train,max_depth,min_size)
    predicted_list,score=test_func(tree,test)
    print(score)
    return predicted_list


# In[238]:


def Bagging_Tree(train,test,max_depth,min_size,times):
    train=pd.DataFrame(train)
    score_list=[]
    predictions_bagging=[]
    for i in range(times):
        sample=train.sample(frac=0.1,replace=True)
        sample=sample.values.tolist()
        predicted_temp=(Decision_Tree_regressor(sample,test,5,20))
        predictions_bagging.append(predicted_temp)
    predictions_bagging=np.array(predictions_bagging)
    final_predictions=np.mean(predictions_bagging,axis=0)
    #print(len(final_predictions))
    actual=[]
    for row in test:
        actual.append(row[4])
        
    #print(len(final_predictions))
    mse=accuracy_metric_func(actual,final_predictions)
    return (mse)


# In[239]:


def random_forest_func(train,test,max_depth,min_size,times):
    score_list=[]
    predictions_bagging=[]
    for i in range(times):
        p=4
        train=pd.DataFrame(X_train)
        train=train.sample(p,axis=1)
        train.insert(4,"label",Y_train,True)
        train.columns=[0,1,2,3,4]
        
        test=pd.DataFrame(X_test)
        test=test.sample(p,axis=1)
        test.insert(4,"label",Y_test,True)
        test.columns=[0,1,2,3,4]
#         train=train.values.tolist()
        test=test.values.tolist()
        
        sample=train.sample(frac=0.1,replace=True)
        sample=sample.values.tolist()
        predicted_temp=(Decision_Tree_regressor(sample,test,5,20))
        predictions_bagging.append(predicted_temp)
    predictions_bagging=np.array(predictions_bagging)
    final_predictions=np.mean(predictions_bagging,axis=0)
    #print(len(final_predictions))
    actual=[]
    for row in test:
        actual.append(row[4])

    #print(len(final_predictions))
    mse=accuracy_metric_func(actual,final_predictions)
    return mse
    


# In[242]:


max_depth=5
min_size=1
times=20
scores=[]
depth=[]
# for max_depth in range(1,16,2):
#     depth.append(max_depth)
#     scores.append(Bagging_Tree(train,test,max_depth,min_size,times))
temp=Decision_Tree_regressor(train,test,max_depth,min_size)
score1=Bagging_Tree(train,test,max_depth,min_size,times)
score2=random_forest_func(train,test,max_depth,min_size,times)
print(score1)
print(score2)


# In[243]:


# plt.plot(depth,scores)
# plt.title("Depth vs Accuracy score")
# plt.xlabel("Depth ")
# plt.ylabel("Accuracy")
# plt.show()
# print(scores)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




