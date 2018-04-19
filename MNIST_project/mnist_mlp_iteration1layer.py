# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(confusionmatrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        confusionmatrix = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusionmatrix)

    plt.imshow(confusionmatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        plt.text(j, i, format(confusionmatrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusionmatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    


if __name__ == "__main__":
    max_iter = 2
    err = np.zeros(max_iter-1)
    arr = np.zeros(max_iter-1)
    iteration = np.zeros(max_iter-1)
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    X_train = X_train.reshape(60000, 784)
    x_train = np.zeros((60000,392))
    x_test = np.zeros((10000,392))
    for i in range(0,392):
        x_train[:,i] = X_train[:,i*2]
        x_test[:,i] = X_test[:,i*2]

    for i in range (1, max_iter):
        print(i)
        batch_size = 100
        nb_classes = 10
        nb_epoch = i   # 20

        # the data, shuffled and split between train and test sets
        X_valid = x_train[50000:,:]
        X_train = x_train[:50000,:]
        y_valid = y_train[50000:]
        y_train = y_train[:50000]
        
        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = x_test.astype('float32')
        X_train /= 255
        X_valid /= 255
        X_test /= 255
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
    
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        model = Sequential()
        model.add(Dense(200, input_shape=(392,))) #dense 512
        model.add(Activation('relu'))
        # model.add(Dropout(0.2)) #0.2
        model.add(Dense(200))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2)) #0.2
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.summary()
        sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', 
                  optimizer='sgd',
                  metrics=['accuracy'])

        history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        verbose=1,
                        validation_data=(X_valid, Y_valid)) # X_test, Y_test

        score = model.evaluate(X_valid, Y_valid, verbose=0) # X_test, Y_test

        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        err[i-1] = score[0]
        arr[i-1] = score[1] * 100
        iteration[i-1] = i
    print(np.argmin(err)) 
    print(np.argmax(arr))    
    plt.figure(1)
    plt.plot(iteration, err,'b')
    plt.title(("Loss with differernt iteration"),fontsize=15)
    plt.xlabel(('Iteration'),fontsize=15)
    plt.ylabel(('Loss'),fontsize=15)
    plt.xlim(1, max_iter - 1)
    plt.show
		
    plt.figure(2)
    plt.plot(iteration, arr,'b')
    plt.title(("Accuracy with differernt iteration"),fontsize=15)
    plt.xlabel(('Iteration'),fontsize=15)
    plt.ylabel(('Accuracy'),fontsize=15)
    plt.xlim(1, max_iter - 1)
    plt.ylim(80, 100)
    plt.show
    
    Y_pred = model.predict(X_test,verbose=2)
    y_pred = np.argmax(Y_pred,axis=1)
    # print Y_pred.shape
    for ix in range(10):
        print (ix, confusion_matrix(np.argmax(Y_test,axis=1), y_pred)[ix].sum())
    print (confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

    class_names = ['0','1','2','3','4','5','6','7','8','9',]
    plt.figure(3)
    plot_confusion_matrix(confusion_matrix(np.argmax(Y_test,axis=1), y_pred), classes=class_names)