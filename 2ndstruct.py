import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import *
import numpy as np
from numpy import array
import pickle as pick


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pick.load(fo)
    return dict
    
def pickle(data,filename):
    f = open(filename,"wb")
    pick.dump(data,f,protocol=2)
    f.close()
    
def compileDictionary(filename):
    xDict=[]
    yDict=[]
    with open(filename) as infile:
        for index,line in enumerate(infile):
            if(index%4==1):
                xDict+=list(line.rstrip('\n'))
            elif(index%4==2):
                yDict+=list(line.rstrip('\n'))
    return xDict,yDict

def convertToIntArray(string,dict):
    output = []
    for i in string:
        output+=[dict[i]]
    return output

def compileNumericalData(filename):
    xDict={'R':0, 'X':1, 'H':2, 'K':3, 'E':4, 'S':5, 'I':6, 'M':7, 'C':8, 'D':9, 'T':10, 'Y':11, 'F':12, 'N':13, 'P':14, 'W':15, 'V':16, 'G':17, 'L':18, 'Q':19, 'A':20}
    yDict={'H':0, 'E':1, 'C':2}
    x=[]
    y=[]
    with open(filename) as infile:
        for index,line in enumerate(infile):
            if(index%4==1):
                x+=[convertToIntArray(line.rstrip('\n'),xDict)]
            elif(index%4==2):
                y+=[convertToIntArray(line.rstrip('\n'),yDict)]
    return x,y
        
def cropWindowData(inputArray,windowSize):
    output = []
    dim = len(inputArray)
    stride = 6
    m=0
    while m+windowSize <= dim:
        output.append(inputArray[m:m+windowSize])
        m+=stride
    return output


def getOneHotEncoding(array):
    output = []
    for x in array:
        output += [list(keras.utils.to_categorical(x, 21))]
    return output

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def generatePSSMInput():
    #trainseq = 1180
    testseq  = 126
    x_pssm=[] 
    for i in range(0,testseq):
        with open('test_pssm/test'+str(i)+'.pssm') as infile:
            for index,line in enumerate(infile):
                lineArray = line.rstrip('\n').split()
                try:
                    if(len(lineArray)>0 and int(lineArray[0])==index-2):
                        x_pssm.append([sigmoid(int(x)) for x in lineArray[2:]])
                except ValueError:
                    x=0
    #pickle(x_pssm,'xtest_pssm')
    return x_pssm

usePSSM = True
windowSize = 30

xte = 0
if(usePSSM):
    xte = unpickle('xtest_pssm')
else:
    xte = unpickle('x_test')
yte = unpickle('y_test')
x_test = []
for x in xte:
    croppedData=cropWindowData(x,windowSize)
    x_test+=croppedData
y_test = []
for y in yte:
    croppedData=cropWindowData(y,windowSize)
    y_test+=croppedData
x_test = array(x_test)
y_test = array(y_test)


xte = 0
if(usePSSM):
    xte = unpickle('xtrain_pssm')
else:
    xte = unpickle('x_train')

yte = unpickle('y_train')

validateLen = []
x_train = []
for x in xte:
    validateLen+=[len(x)]
    croppedData=cropWindowData(x,windowSize)
    x_train+=croppedData
y_train = []
for index,y in enumerate(yte):
    if(validateLen[index]!=len(y)):
        print("Index: "+str(index))
        print("X: "+str(validateLen[index]))
        print("Y: "+str(len(y)))
    croppedData=cropWindowData(y,windowSize)
    y_train+=croppedData
x_train = array(x_train)
y_train = array(y_train)
print(len(x_train))
print(len(y_train))

##########################################
###### MODEL 2 #######
###########################################

# model=Sequential()
# model.add(Conv1D(128, 11, strides=1, padding='same', input_shape=(windowSize, 21), activation="relu"))
# model.add(Dropout(0.4))
# model.add(Conv1D(64, 11, strides=1, padding='same', activation="relu"))
# model.add(Dropout(0.4))
# model.add(Conv1D(3, 11, strides=1, padding='same', activation="softmax"))
#
# print(model.summary())
# model.compile(loss='mse',
#         optimizer='sgd',
#         metrics=['accuracy'])
        
##########################################
###### MODEL 1 #######
###########################################
#Define the model
inputDim=0
if(usePSSM):
    inputDim = 20
else:
    inputDim = 21
model = Sequential()
model.add(Conv1D(32, kernel_size=15, strides= 1,
                 activation='relu',padding='same',
                 input_shape=(windowSize,inputDim)))
model.add(Dropout(0.25))
model.add(Conv1D(64, kernel_size=7  , strides= 1,
                 activation='relu', padding='same'))
# model.add(MaxPooling1D(pool_size=1, padding='valid'))
model.add(Dropout(0.50))
model.add(Conv1D(128, kernel_size=5  , strides= 1,
                 activation='relu', padding='same'))
# model.add(MaxPooling1D(pool_size=1, padding='valid'))
model.add(Dropout(0.50))

model.add(Conv1D(256, kernel_size=3  , strides= 1,
                 activation='relu', padding='same'))
# model.add(MaxPooling1D(pool_size=1, padding='valid'))
model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.50))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.50))
# model.add(Dense(windowSize*3, activation='relu'))
# model.add(Reshape((windowSize,3)))
model.add(Conv1D(3, windowSize, strides=1,padding='same', activation="softmax"))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
        
print(model.summary())
model.fit(x_train[:1062], y_train[:1062], epochs=15, batch_size=30,validation_data = (x_train[1062:], y_train[1062:]))


prediction = model.predict(x_test, batch_size=30)
