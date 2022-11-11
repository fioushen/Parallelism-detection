import tensorflow as tf
import numpy as np
from numpy import *
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import torch



def data_conversion(output):
 test_output=[]
 for value in output:
  test_output.append(tf.convert_to_tensor(value))
 return test_output

#精确率评价指标
def metric_precision(y_true,y_pred):
 TP=tf.reduce_sum(y_true*tf.round(y_pred))
 TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
 FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
 FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
 precision=TP/(TP+FP)

 return precision

#召回率评价指标
def metric_recall(y_true,y_pred):
 TP=tf.reduce_sum(y_true*tf.round(y_pred))
 TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
 FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
 FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
 recall=TP/(TP+FN)
 return recall

#F1-score评价指标
def metric_F1score(y_true,y_pred):
 TP = 0
 TN = 0
 FP = 0
 FN = 0
 for i in range(len(y_true)):
  if y_pred[i] == 1:
   if y_pred[i] - y_true[i] == 0:
    TP += 1
   elif y_pred[i] - y_true[i] == 1:
    FN += 1  # 将可以并行的预测成了不可并行的
  if y_pred[i] == 0:
   if y_pred[i] - y_true[i] == 0:
    TN += 1
   elif y_true[i] - y_pred[i] == 1:
    FP += 1

 precision=TP/(TP+FP)
 recall=TP/(TP+FN)
 F1score=2*precision*recall/(precision+recall)
 return precision, recall, F1score














