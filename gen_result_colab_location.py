from partialdischarge_funcs import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint,CSVLogger
from itertools import count
import numpy as np
import pylab as pl
import pandas as pd
import keras,os

randSeed=10
testSize=.01
nParams=50

###########################
# Location 4
###########################

nfiltbases=np.random.choice(range(50,150),nParams)
filtlens=np.random.choice(range(6,12),nParams)
lrs=np.random.choice([0.01,.005,.001,.0005,.0001],nParams)
dropProbs=np.random.choice([.3,.4,.5,.6,.7,],nParams)

params=zip(nfiltbases,filtlens,lrs,dropProbs)
bestsofar=0


bestsofar=0
counter=count(1)


for param in params:

  # Looping through the 3 1-v-all cases
  withinscores=[]
  crossscores=[]
  for targClass in range(3):

    loc=CreateLocationData()
    X4,X9,Y4,Y9=[loc[tmp] for tmp in ['X4','X9','Y4','Y9']]

    # Convert to 1-v-all?
    ax1=targClass
  
    foo=[0,1,2]
    foo.remove(targClass)
    ax2a=foo[0]
    ax2b=foo[1]
    Y4=np.concatenate((Y4[:,ax1].reshape((-1,1)),np.logical_or(Y4[:,ax2a],Y4[:,ax2b]).reshape((-1,1))),axis=1)
    Y9=np.concatenate((Y9[:,ax1].reshape((-1,1)),np.logical_or(Y9[:,ax2a],Y9[:,ax2b]).reshape((-1,1))),axis=1)
    
    xtrain,xtest,ytrain,ytest=train_test_split(X4,Y4,test_size=testSize,random_state=randSeed)
    xtrain9,xtest9,ytrain9,ytest9=train_test_split(X9,Y9,test_size=testSize,random_state=randSeed)
    
    cnn=CreateCNN((100,150,150,150,200),(10,10,10,10,10),poolSizes=(3,3,3,3,3),lr=.0001,dropProb=.7,reg=.0,nOutputs=2,activation=keras.layers.advanced_activations.LeakyReLU)
    callbacks=[keras.callbacks.ModelCheckpoint('bestMod4_class'+str(targClass)+'.hdf',save_best_only=True,monitor='val_loss'),keras.callbacks.CSVLogger('trainres4.csv')]

    cnn.fit(xtrain,ytrain,epochs=100,callbacks=callbacks, validation_split=.05) # validation_data=(xtest9,ytest9)
    cnnbest=keras.models.load_model('bestMod4_class'+str(targClass)+'.hdf')
  
    print("Results using class"+str(targClass)+" as target Class")
    withinscore=np.sum(cnnbest.predict(xtest).argmax(axis=1)==ytest.argmax(axis=1))/len(ytest)
    crossscore=np.sum(cnnbest.predict(xtrain9).argmax(axis=1)==ytrain9.argmax(axis=1))/len(ytrain9)
    
    withinscores.append(withinscore)
    crossscores.append(crossscore)

  tmpacc=np.mean(crossscores)
  if tmpacc>bestsofar:
    bestsofar=tmpacc
    print("New top (4->9): "+str(bestsofar))
    for targClass in range(3):
      os.system('mv bestMod4_class'+str(targClass)+'.hdf overallBest4_class'+str(targClass)+'.hdf')
    open("bestParams4.txt","w+").write(str(param)+', acc:'+str(bestsofar))


  
###########################
# Now, location 9
###########################

nfiltbases=np.random.choice(range(50,150),nParams)
filtlens=np.random.choice(range(6,12),nParams)
lrs=np.random.choice([0.01,.005,.001,.0005,.0001],nParams)
dropProbs=np.random.choice([.3,.4,.5,.6,.7,],nParams)

params=zip(nfiltbases,filtlens,lrs,dropProbs)
bestsofar=0


bestsofar=0
counter=count(1)


for param in params:

  # Looping through the 3 1-v-all cases
  withinscores=[]
  crossscores=[]
  for targClass in range(3):

    loc=CreateLocationData()
    X4,X9,Y4,Y9=[loc[tmp] for tmp in ['X4','X9','Y4','Y9']]

    # Convert to 1-v-all?
    ax1=targClass
  
    foo=[0,1,2]
    foo.remove(targClass)
    ax2a=foo[0]
    ax2b=foo[1]
    Y4=np.concatenate((Y4[:,ax1].reshape((-1,1)),np.logical_or(Y4[:,ax2a],Y4[:,ax2b]).reshape((-1,1))),axis=1)
    Y9=np.concatenate((Y9[:,ax1].reshape((-1,1)),np.logical_or(Y9[:,ax2a],Y9[:,ax2b]).reshape((-1,1))),axis=1)
    
    xtrain,xtest,ytrain,ytest=train_test_split(X4,Y4,test_size=testSize,random_state=randSeed)
    xtrain9,xtest9,ytrain9,ytest9=train_test_split(X9,Y9,test_size=testSize,random_state=randSeed)
    
    cnn=CreateCNN((100,150,150,150,200),(10,10,10,10,10),poolSizes=(3,3,3,3,3),lr=.0001,dropProb=.7,reg=.0,nOutputs=2,activation=keras.layers.advanced_activations.LeakyReLU)
    callbacks=[keras.callbacks.ModelCheckpoint('bestMod9_class'+str(targClass)+'.hdf',save_best_only=True,monitor='val_loss'),keras.callbacks.CSVLogger('trainres9.csv')]

    cnn.fit(xtrain9,ytrain9,epochs=100,callbacks=callbacks, validation_split=.05) # validation_data=(xtest9,ytest9)
    cnnbest=keras.models.load_model('bestMod9_class'+str(targClass)+'.hdf')
  
    print("Results using class"+str(targClass)+" as target Class")
    withinscore=np.sum(cnnbest.predict(xtest9).argmax(axis=1)==ytest9.argmax(axis=1))/len(ytest9)
    crossscore=np.sum(cnnbest.predict(xtrain).argmax(axis=1)==ytrain.argmax(axis=1))/len(ytrain)
    
    withinscores.append(withinscore)
    crossscores.append(crossscore)

  tmpacc=np.mean(crossscores)
  if tmpacc>bestsofar:
    bestsofar=tmpacc
    print("New top (9->4): "+str(bestsofar))
    for targClass in range(3):
      os.system('mv bestMod9_class'+str(targClass)+'.hdf overallBest9_class'+str(targClass)+'.hdf')
    open("bestParams9.txt","w+").write(str(param)+', acc:'+str(bestsofar))
