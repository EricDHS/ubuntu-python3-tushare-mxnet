# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)

def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Flatten())
        #net.add(gluon.nn.Dense(256, activation="sigmoid"))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

def Train(net, X_train, y_train, X_test, y_test, epochs,
          verbose_epoch, learning_rate, weight_decay):
    train_loss = []
    if X_test is not None:
        test_loss = []
    batch_size = 99 
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size,shuffle=True)
    #trainer = gluon.Trainer(net.collect_params(), 'adam',
    #                        {'learning_rate': learning_rate,
    #                         'wd': weight_decay})
    trainer = gluon.Trainer(net.collect_params(), 'adadelta', {'rho': 0.9999})
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

            cur_train_loss = get_rmse_log(net, X_train, y_train)
        #if epoch > verbose_epoch:
        #    print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
        #if (cur_train_loss < 0.04):
        preds = net(X_test_new).asnumpy()
        tmp_value = pd.Series(preds.reshape(1, -1)[0]).values[0]
        print(tmp_value)
        cur_test_loss = get_rmse_log(net, X_test, y_test)
        print("Epoch {}, train loss: {} test loss: {}".format(epoch, cur_train_loss, cur_test_loss))
        train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss

def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_loss, test_loss = Train(net, X_val_train, y_val_train, X_val_test, y_val_test, epochs, verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        preds = net(X_test_new).asnumpy()
        print(pd.Series(preds.reshape(1, -1)[0]))
        test_loss_sum += test_loss
    preds = net(X_test_new).asnumpy()
    print(pd.Series(preds.reshape(1, -1)[0]))
    return train_loss_sum / k, test_loss_sum / k

def learn(epochs, verbose_epoch, X_train, y_train, learning_rate,
          weight_decay):
    net = get_net()
    Train(net, X_train, y_train, None, None, epochs, verbose_epoch,
          learning_rate, weight_decay)
    preds = net(X_test).asnumpy()
    print(pd.Series(preds.reshape(1, -1)[0]))

def predict(filename):
    global X_test_new
    train = pd.read_csv('data/mxnet/{}'.format(filename))
    all_X = train.loc[:, 'open':'turnover']
    numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
    all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))

    all_X = pd.get_dummies(all_X, dummy_na=True)

    all_X = all_X.fillna(all_X.mean())

    num_train = train.shape[0]

    a = pd.DataFrame()
    fields = 1
    #for i in range(fields):
    #    a = pd.concat([a, all_X[:num_train]], axis=1)

    #all_X = a
    X_train = all_X[:num_train].as_matrix()
    y_train = train.predict.as_matrix()
    X_train = nd.array(X_train).reshape((num_train, fields, 1, 14))
    y_train = nd.array(y_train)

    y_train.reshape((num_train, 1))
    #X_test = nd.array(X_test)
    X_test = all_X[num_train-1:].as_matrix()
    X_test = nd.array(X_test).reshape((1, fields, 1, 14))

    X_test_new = all_X[num_train-1:].as_matrix()
    X_test_new = nd.array(X_test_new).reshape((1, fields, 1, 14))


    k_fold_cross_valid(k, epochs, verbose_epoch, X_train,
                                               y_train, learning_rate, weight_decay)
square_loss = gluon.loss.L2Loss()
k = 5
epochs = 300
verbose_epoch = 0
learning_rate = 0.2
weight_decay = 0.0
X_test_new = 0

print("Open")
predict('open-002652.csv')

print("Close")
predict('close-002652.csv')
print("high")
predict('high-002652.csv')
print("low")
predict('low-002652.csv')
