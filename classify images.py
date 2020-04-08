#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import pandas as pd
import numpy as np
import statistics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statistics
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys


# In[3]:


class LogisticRegression:
    all_images = pd.DataFrame([])
    scale_factor = 60
    all_images_std = pd.DataFrame([])
    labels_frm = []
    mean_std = list()
    theta_val = []
    cost_val = []
    alpha = 0.005
    iterations = 1000
    classes = None
    principle_comp = None
    
    def find_mean_std(self):
        for cols in range(len(self.all_images.columns)):
            col = self.all_images.iloc[:,cols]
            mean_val = statistics.mean(col)
            std_val = statistics.stdev(col)
            self.mean_std.append([mean_val,std_val])
            
    def preproccess_data(self,mat):
        for i in range(len(self.mean_std)):
            mat[i] = (mat[i]-self.mean_std[i][0])/self.mean_std[i][1]
        return mat
    
    def collect_data(self,file_location):
        file_read = open(file_location, 'r')
        file_lines = file_read.readlines();
        img_shape = None
        i = 0 
        for lines in file_lines:
            img_name = lines.split(' ')[0]
            img_label = lines.split(' ')[1]
            img = cv2.imread(img_name,0)
            self.labels_frm.append(img_label.strip())
            img_width = 64
            img_height = 64
            resized_img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
            img_shape = resized_img.shape
            img_flatten = pd.Series(resized_img.flatten(), name=i)
            i+=1
            self.all_images = self.all_images.append(img_flatten)
        return self.labels_frm

    def standardize_data(self):
        self.find_mean_std()
        self.all_images_std = self.preproccess_data(self.all_images)
    
    def performPCA(self):
        eigen_vectors, s, v = np.linalg.svd(self.all_images_std.T)
        self.principle_comp = eigen_vectors[:,:300]
        predicted_img = self.all_images_std.dot(self.principle_comp)
        return predicted_img
        
    def train_validation_split(self,validation_data_size):
        if isinstance(validation_data_size, float):
            validation_data_size=round(validation_data_size * len(self.all_images_std))
        indices = [i for i in range(0,self.all_images_std.shape[0])]
        valid_indices = random.sample(indices, validation_data_size)
        valid_data = self.all_images_std.loc[valid_indices]
        train_data = self.all_images_std.drop(valid_indices)
        valid_labels = [self.labels_frm[i] for i in valid_indices]
        train_labels = [self.labels_frm[i] for i in indices if i not in valid_indices]
        return train_data, train_labels, valid_data, valid_labels
    
    def calculate_sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    
    def find_gradient_descent(self,train_data,h,theta,c_labels):
        m = len(c_labels)
        j = (train_data.T.dot(h-c_labels))/m
        theta -= self.alpha * j
        return theta
        
    def run_logistic_regression(self,train_data,train_labels):
        rows,cols = train_data.shape
        train_data = np.c_[np.ones(rows), train_data]
        self.classes = np.unique(train_labels)
        for c in self.classes:
            c_labels = np.where(train_labels==c, 1, 0)
            theta = np.zeros(train_data.shape[1])
            for i in range(self.iterations):
                z = train_data.dot(theta)
                h = self.calculate_sigmoid(z)
                theta = self.find_gradient_descent(train_data,h,theta,c_labels)
            self.theta_val.append((theta, c))

    
    def PCAtransform(self,mat):
        return mat.dot(self.principle_comp)
    
    def predict(self,predict_file_location):
        predict_datafrm = pd.DataFrame([])
        file_read = open(predict_file_location, 'r')
        file_lines = file_read.readlines();
        i = 0
        for lines in file_lines:
            img_name = lines.split(' ')[0].strip()
            img = cv2.imread(img_name,0)
            img_width = 64
            img_height = 64
            resized_img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
            img_flatten = pd.Series(resized_img.flatten(), name=i)
            i+=1
            predict_datafrm = predict_datafrm.append(img_flatten)
        predict_stand = self.preproccess_data(predict_datafrm)
        predict_PCA = self.PCAtransform(predict_stand)
        predict_input = predict_PCA.values
        rows,cols = predict_input.shape
        predict_input = np.c_[np.ones(rows), predict_input]
        predicted_values = [max((self.calculate_sigmoid(i.dot(theta)),c) for theta, c in self.theta_val)[1] for i in predict_input]
        return predicted_values

lr = LogisticRegression()
file_location = sys.argv[1]
# file_location = './sample_train_2.txt'
train_labels = lr.collect_data(file_location)
lr.standardize_data()
train_data = lr.performPCA()
lr.run_logistic_regression(train_data.values,np.array(train_labels))
# predict_file_location = './sample_test_2.txt'
predict_file_location = sys.argv[2]
predicted_labels = lr.predict(predict_file_location)
for val in predicted_labels:
    print(val)


# In[ ]:




