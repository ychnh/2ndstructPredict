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
import math


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pick.load(fo)
    return dict
    
def pickle(data,filename):
    f = open(filename,"wb")
    pick.dump(data,f,protocol=2)
    f.close()
    
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
trainseq = 1180
testseq  = 126
x_pssm=[] 
for i in range(0,trainseq):
    sequence=[]
    with open('train_pssm/train'+str(i)+'.pssm') as infile:
        for index,line in enumerate(infile):
            lineArray = line.rstrip('\n').split()
            try:
                if(len(lineArray)>0 and int(lineArray[0])==index-2):
                    sequence.append([sigmoid(int(x)) for x in lineArray[2:]])
            except ValueError:
                x=0
    x_pssm.append(sequence)
pickle(x_pssm,'xtrain_pssm')
#print(x_pssm[0])