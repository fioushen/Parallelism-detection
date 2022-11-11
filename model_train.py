import astnn_predict as astnn
import dgcnn_predict_CFG as dgcnn_cfg
import dgcnn_predict_DFG as dgcnn_dfg
import dgcnn_predict as dgcnn
import numpy as np
import time
import tensorflow as tf
from threading import Thread
from time import sleep
import sys




print('======================DGCNN预测=================================')
#
start = time.time()
dgcnn_accuracy_cfg, dgcnn_output_list_cfg, dgcnn_test_output_cfg = dgcnn_cfg.train('sigmoid',1)
end = time.time()
print('dgcnn_cfg的预测准确率是：', dgcnn_accuracy_cfg)
print('dgcnn_cfg的预测结果是：',dgcnn_test_output_cfg)
print('dgcnn_cfg的预测时间是：', end-start)

start1 = time.time()
dgcnn_accuracy_dfg, dgcnn_output_list_dfg, dgcnn_test_output_dfg = dgcnn_dfg.train('sigmoid',1)
end1 = time.time()
print('dgcnn_dfg的预测准确率是：', dgcnn_accuracy_dfg)
print('dgcnn_dfg的预测结果是：',dgcnn_test_output_dfg)
print('dgcnn_dfg的预测时间是：', end1-start1)

start3 = time.time()
dgcnn_accuracy_xfg, dgcnn_output_list_xfg, dgcnn_test_output_xfg = dgcnn.train('sigmoid',1)
end3 = time.time()
print('dgcnn_dfg的预测准确率是：', dgcnn_accuracy_dfg)
print('dgcnn_dfg的预测结果是：',dgcnn_test_output_dfg)
print('dgcnn_dfg的预测时间是：', end3-start3)


print('======================ASTNN预测=================================')


start2 = time.time()
astnn_accuracy,astnn_output_list, astnn_test_output = astnn.astnn_prediction()
end2 = time.time()
print('astnn的预测准确率是：', astnn_accuracy)
print('astnn的预测结果是：', astnn_test_output)
print('astnn的预测时间是：', end2-start2)



label = []


x1 = np.array(astnn_test_output) * d + np.array(dgcnn_test_output_cfg) * a + np.array(dgcnn_test_output_dfg) * b + np.array(dgcnn_test_output_xfg) * c

pred_list1 = np.argmax(x1, axis=1)
print(pred_list1)
acc1 = 0
for i in range(len(label)):
    if pred_list1[i] - label[i] == 0:
        acc1 += 1


y_true = label
y_pred = pred_list
precision, recall, F1score = metric_F1score(y_true,y_pred)
print("precision的值为：", precision)
print("recall的值为：", recall)
print("F1的值为：", F1score)









