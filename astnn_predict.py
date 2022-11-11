import pandas as pd
import random
import torch
import time
import math
import numpy as np
# import utils.astnn_util.pipeline as data_prepare
from gensim.models.word2vec import Word2Vec
from model_attention import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys



def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        # labels.append(item[2] - 1)
        labels.append(item[2])
    return data, torch.LongTensor(labels)


# if __name__ == '__main__':
def astnn_prediction():
    root = data_path

    data = pd.read_pickle(root + 'train/blocks.pkl')
    print(data)
    data = data.sample(frac=1)
    train_data = data.iloc[:int(800)]
    print('len(train_data) = ', len(train_data))
    val_data = data.iloc[int(800):int(1000)]
    print('len(val_data) = ', len(val_data))
    test_data = val_data


    word2vec = Word2Vec.load(root + "train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2

    num_filters = 128

    EPOCHS = 50
    BATCH_SIZE = 32
    USE_GPU = False
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    print(embeddings)
    print(len(embeddings[0]))

    model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, num_filters,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    test_loss = []
    train_acc_ = []
    val_acc_ = []
    test_acc = []


    best_acc = 0.0
    print('Start training...')
    # training procedure



    for epoch in range(EPOCHS):

        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        model.train()
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()


            output = model(train_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        test_output = []
        predict_output = []
        model.eval()
        while i < len(val_data):
            batch = get_batch(val_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_labels = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs)
            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item() * len(val_inputs)



            output_tensor_tolist = output.detach().numpy()
            for output in output_tensor_tolist:
                test_output.append(output)

            predicted_tensor_tolist = np.array(predicted).tolist()
            for value in predicted_tensor_tolist:
                predict_output.append(value)



        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc.item() / total > best_acc:
            best_acc = total_acc.item() / total


            torch.save(model.state_dict(), 'model/model_102_%s_%s.pkl' % (epoch, best_acc))
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0


    model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                                   num_filters,USE_GPU, embeddings)
    model.load_state_dict(torch.load(('model.pkl')))

    model.eval()
    predict_output = []
    test_labels_list = []
    test_output = []
    acc = 0
    unpar_to_par = 0
    par_to_unpar = 0
    # print(test_data)
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)


        loss = loss_function(output, Variable(test_labels))


        _, predicted = torch.max(output.data, 1) #0是每列的最大值吧，1是每行的最大值
        output_tensor_tolist = output.detach().numpy()
        # output_tensor_tolist = np.detach.array(output).tolist()
        for output in output_tensor_tolist:
            test_output.append(output)






        # print(test_labels)
        test_labels_tensor_tolist = np.array(test_labels).tolist()
        for label in test_labels_tensor_tolist:
            test_labels_list.append(label)


        predicted_tensor_tolist = np.array(predicted).tolist()
        for value in predicted_tensor_tolist:
            predict_output.append(value)

        total_acc += (predicted == test_labels).sum()
        # print(total_acc)  # 预测准确的总数
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    return best_acc



predict_value = astnn_prediction()

print('astnn的预测结果是：', predict_value)


