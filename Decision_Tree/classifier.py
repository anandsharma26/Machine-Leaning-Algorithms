#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # New Section

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.setrecursionlimit(2500)
from scipy import stats


# In[108]:


#path=pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv")
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

X1=data.values.tolist()
X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
train=[]
test=[]
for row in X1:
    if(row[0]==2010 or row[0]==2012):
        X_train.append(row)
        Y_train.append(row[1])
        train.append(row)
    elif(row[0]==2011 or row[0]==2013):
        X_test.append(row)
        Y_test.append(row[1])
        test.append(row)

X_train=np.array(X_train)
X_train=np.delete(X_train,1,1)
X_train=(X_train[:,:]).tolist()

X_test=np.array(X_test)
X_test=np.delete(X_test,1,1)
X_test=(X_test[:,:]).tolist()

# print(len(X_train))
# print(len(X_test))
# print(len(Y_train))
# print(len(Y_test))
# print(len(train))
# print(len(test))
# #print(X_train)


# In[46]:


def group_split(index,value,dataset):
    left=list()
    right=list()
    for row in dataset:
        if row[index]>value:
            right.append(row)
        else:
            left.append(row)
    return left,right


# In[100]:


def calculate_gini(groups,classses):
    left,right=groups[0],groups[1]

    total_len=len(left)+len(right)
    if(len(left)==0):
        loc_score=0.0
    else:
        loc_score=0.0
        for clas in classses:
            temp=float([i[1] for i in left].count(clas))/len(left)
            loc_score+=temp*temp
        loc_score=1-loc_score
    if (len(right)==0):
        loc_score1=0.0
    else:
        loc_score1=0.0
        for clas in classses:
            temp1=[i[1] for i in right].count(clas)/len(right)
            loc_score1+=temp1*temp1
        loc_score1=1-loc_score1
    gini=(len(left)/total_len)*(loc_score)+(len(right)/total_len)*(loc_score1)
    return gini


# In[106]:


def get_best_split(X):
    classes=list(set(row[1] for row in X))
    #print(classes)
    b_index,b_value,b_score,b_groups=None,None,10000000,None
    for j in range(len(X[0])):
        if j==1:
            continue
        feature_list=[row[j] for row in X]
        unique_vals=set(feature_list)
        for value in unique_vals:
            groups=group_split(j,value,X)
            gini=calculate_gini(groups,classes)
            #print("i th row and j th column",gini)
            if(gini<b_score):
                b_score=gini
                b_groups=groups
                b_index=j
                b_value=value
    #print(b_index,b_value,b_score,b_groups)
    return {'index':b_index,'value':b_value,'groups':b_groups}


# In[55]:


def terminal_node(root):
    outcomes = [row[1] for row in root]
    mode=stats.mode(outcomes)[0][0]
    return mode
    #return max(set(outcomes), key=outcomes.count) if outcomes else None


# In[56]:


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


# In[114]:


def decision_tree_Build(train,maxdepth,min_size):
    root=get_best_split(train)
    split_func(root,maxdepth,min_size,1)
    return root


# In[115]:


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
            


# In[116]:


def accuracy_metric_func(actual,predicted):
    count=0
    n=len(actual)
    for i in range(n):
        if actual[i]==predicted[i]:
            count+=1
    return (float(count/n)*100.0)


# In[133]:


def Decision_Tree_classifier_func(train,test,max_depth,min_size):
    tree=decision_tree_Build(train,5,1)
    predictions_list=[]
    actual=[]
    for row in test:
        actual.append(row[1])
        predictions_list.append(predict(tree,row))

    score=accuracy_metric_func(actual,predictions_list)
    #print(score)
    return predictions_list


# In[134]:


def random_forest_func(train,test,max_depth,min_size,times):
    score_list=[]
    predictions_bagging=[]
    for i in range(times):
        p=4
        train=pd.DataFrame(X_train)
        train=train.sample(p,axis=1)
        train.insert(1,"label",Y_train,True)
        train.columns=[0,1,2,3,4]
        
        test=pd.DataFrame(X_test)
        test=test.sample(p,axis=1)
        test.insert(1,"label",Y_test,True)
        test.columns=[0,1,2,3,4]
       # print(test)
#         train=train.values.tolist()
        test=test.values.tolist()
        
        sample=train.sample(frac=0.1,replace=True)
        sample=sample.values.tolist()
        predicted_temp=(Decision_Tree_classifier_func(sample,test,5,20))
        predictions_bagging.append(predicted_temp)
    predictions_bagging=np.array(predictions_bagging)
    final_predictions=stats.mode(predictions_bagging)
    final_predictions=final_predictions[0][0].tolist()
    #print(len(final_predictions))
    actual=[]
    for row in test:
        actual.append(row[1])
    
    mse=accuracy_metric_func(actual,final_predictions)
    return (mse)
    #print((actual))


# In[135]:



def bagging_tree(train,test,max_depth,min_size,times):
    
    train=pd.DataFrame(train)
    score_list=[]
    predictions_bagging=[]
    for i in range(times):
        sample=train.sample(frac=0.1,replace=True)
        sample=sample.values.tolist()
        predicted_temp=(Decision_Tree_classifier_func(sample,test,max_depth,min_size))
        predictions_bagging.append(predicted_temp)
        
    predictions_bagging=np.array(predictions_bagging)
    final_predictions=stats.mode(predictions_bagging)
    final_predictions=final_predictions[0][0]
    #print(final_predictions)
    #print(len(final_predictions))
    actual=[]
    for row in test:
        actual.append(row[1])
        
    #print(len(final_predictions))
    mse=accuracy_metric_func(actual,final_predictions)
    return (mse)
    


# In[140]:


max_depth=5
min_size=10
times=50
scores=[]
depth=[]
# for max_depth in range(1,16,2):
#     depth.append(max_depth)
#     scores.append(random_forest_func(train,test,max_depth,min_size,times))
score1=Decision_Tree_classifier_func(train,test,max_depth,min_size)
score2=bagging_tree(train,test,max_depth,min_size,times)
score3=random_forest_func(train,test,max_depth,min_size,times)
print(score1)
print(score2)
print(score3)


# In[141]:



# plt.plot(depth,scores)
# plt.title("Depth vs Accuracy score")
# plt.xlabel("Depth ")
# plt.ylabel("Accuracy")
# plt.show()
# print(scores)


# In[ ]:





# In[ ]:



    


# In[ ]:




