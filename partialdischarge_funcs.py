from scipy.io import loadmat
from scipy.signal import periodogram
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
import sklearn.qda
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from keras.models import Sequential as sq
from keras.layers import Convolution1D,MaxPooling1D,Flatten,Dense,Dropout,AveragePooling1D,LSTM
import pandas as pd
import numpy as np
import sklearn,re,os,pdb,sys,pylab,keras

def Folder2Matrix(path='.'):

    '''
    Load in all sample*.mat files from a folder and return as an array n x dim
    '''

    matFiles=[os.path.normpath(path+os.path.sep+tmp) for tmp in os.listdir(path) if re.compile(".+\.mat$").match(tmp)]
    timeseries=np.concatenate([np.array(loadmat(tmp)['X']) for tmp in matFiles],axis=1)
    return np.array([timeseries]).T

def CreateCNN(classifier,nfilt1=50,lenfilt1=30,nfilt2=50,lenfilt2=20,nhidden=50,dropkeep=.35,lr=0.008):
    ''' 
    Return conv. NN
    '''

    cnn=sq()
    cnn.add(Convolution1D(nfilt1,lenfilt1,input_shape=(2500,1),strides=1,activation='relu'))
    cnn.add(MaxPooling1D(100,50))
    cnn.add(Dropout(dropkeep))
    cnn.add(Convolution1D(nfilt2,lenfilt2,strides=1,activation="relu"))
    cnn.add(MaxPooling1D(10,5))
    cnn.add(Dropout(dropkeep))

    cnn.add(Flatten())
    cnn.add(Dense(nhidden,activation='relu'))
    cnn.add(Dropout(dropkeep))
    cnn.add(Dense(2,activation="softmax"))
    adam=keras.optimizers.Adam(lr=lr)
    cnn.compile(loss='categorical_crossentropy',optimizer=adam)
    
    return cnn

def CreateLSTM(classifier,nFilters=50,lenFilters=30):
    ''' 
    Return conv. NN
    '''

    cnn=sq()
    cnn.add(LSTM(nFilters,dropout=.5,recurrent_dropout=.5,implementation=2,input_shape=(2500,1),return_sequences=True))
    cnn.add(LSTM(30,dropout=.5,recurrent_dropout=.5,implementation=2,input_shape=(2500,1),return_sequences=False))

    cnn.add(Dense(50,activation='relu'))
    cnn.add(Dropout(.5))
    cnn.add(Dense(2,activation="softmax"))
    adam=keras.optimizers.Adam(lr=0.008)
    cnn.compile(loss='categorical_crossentropy',optimizer=adam)
    
    return cnn

def TensorSplit(x,y,test_size=0.25,random_state=0):
    trainVec=np.array([tmp>test_size for tmp in np.random.rand(len(x),)])
    testVec=~trainVec
    return (x[trainVec],x[testVec],y[trainVec],y[testVec])

def CNNScore(model,testx,testy):
    return (testy[model.predict(testx)>0.5].sum()/float(len(testy)))

def CreateLocationData():

    # Creating the data
    surf4=Folder2Matrix('Surface/PD_Location4')
    sharp4=Folder2Matrix('Sharp/PD_Location4')
    void4=Folder2Matrix('Void/PD_Location4')
    nsurf4=len(surf4)
    nsharp4=len(sharp4)
    nvoid4=len(void4)
    X4=np.concatenate((surf4,sharp4,void4))
    Ysurf4=np.concatenate((np.tile([1,0],(nsurf4,1)),np.tile([0,1],(nsharp4+nvoid4,1))))
    Ysharp4=np.concatenate((np.tile([0,1],(nsurf4,1)),np.tile([1,0],(nsharp4,1)),np.tile([0,1],(nvoid4,1))))
    Yvoid4=np.concatenate((np.tile([0,1],(nsurf4+nsharp4,1)),np.tile([1,0],(nvoid4,1))))
    
    surf9=Folder2Matrix('Surface/PD_Location9')
    sharp9=Folder2Matrix('Sharp/PD_Location9')
    void9=Folder2Matrix('Void/PD_Location9')
    nsurf9=len(surf9)
    nsharp9=len(sharp9)
    nvoid9=len(void9)
    X9=np.concatenate((surf9,sharp9,void9))
    Ysurf9=np.concatenate((np.tile([1,0],(nsurf9,1)),np.tile([0,1],(nsharp9+nvoid9,1))))
    Ysharp9=np.concatenate((np.tile([0,1],(nsurf9,1)),np.tile([1,0],(nsharp9,1)),np.tile([0,1],(nvoid9,1))))
    Yvoid9=np.concatenate((np.tile([0,1],(nsurf9+nsharp9,1)),np.tile([1,0],(nvoid9,1))))

    return dict(X4=X4,X9=X9,Ysurf4=Ysurf4,Ysharp4=Ysharp4,Yvoid4=Yvoid4,Ysurf9=Ysurf9,Ysharp9=Ysharp9,Yvoid9=Yvoid9)
