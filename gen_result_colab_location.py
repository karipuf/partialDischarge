from partialdischarge_funcs import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint,CSVLogger
from itertools import count
import numpy as np
import pylab as pl
import pandas as pd
import keras,os,sys

randSeed=10
testSize=.01
nParams=80


loc=CreateLocationData()
X4,X9,Y4,Y9=[loc[tmp] for tmp in ['X4','X9','Y4','Y9']]

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

   print("(4->9) Running paramset #"+str(next(counter)))
   sys.stdout.flush()
  
   nfiltbase,filtlen,lr,dropProb=param
   nfilts=[int(nfiltbase*tmp) for tmp in [1,1.5,1.5,1.5,2]]   
   cnn=CreateCNN(nfilts,[int(filtlen)]*len(nfilts),poolSizes=(3,3,3,3,3),lr=lr,dropProb=dropProb,reg=.0,nOutputs=3)  
   
   callbacks=[keras.callbacks.ModelCheckpoint('bestMod4.hdf',save_best_only=True,monitor='val_acc'),keras.callbacks.CSVLogger('trainres4.csv')]
  
   cnn.fit(X4,Y4,epochs=100,batch_size=64,callbacks=callbacks, validation_split=.05,verbose=0)

   cnnbest=keras.models.load_model('bestMod4.hdf')
   tmpacc=np.sum(cnnbest.predict(X9).argmax(axis=1)==Y9.argmax(axis=1))/len(Y9)
  
   if tmpacc>bestsofar:
      bestsofar=tmpacc
      print("New top (4->9): "+str(bestsofar))
      os.system('mv bestMod4.hdf overallBest4.hdf')
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

   print("(9->4) Running paramset #"+str(next(counter)))
   sys.stdout.flush()
    
   nfiltbase,filtlen,lr,dropProb=param
   nfilts=[int(nfiltbase*tmp) for tmp in [1,1.5,1.5,1.5,2]]   
   cnn=CreateCNN(nfilts,[int(filtlen)]*len(nfilts),poolSizes=(3,3,3,3,3),lr=lr,dropProb=dropProb,reg=.0,nOutputs=3)  
   
   
   callbacks=[keras.callbacks.ModelCheckpoint('bestMod9.hdf',save_best_only=True,monitor='val_acc'),keras.callbacks.CSVLogger('trainres9.csv')]
   
   cnn.fit(X9,Y9,epochs=100,batch_size=64,callbacks=callbacks, validation_split=.05,verbose=0)
   
   cnnbest=keras.models.load_model('bestMod9.hdf')
   tmpacc=np.sum(cnnbest.predict(X4).argmax(axis=1)==Y4.argmax(axis=1))/len(Y4)
   
   if tmpacc>bestsofar:
      bestsofar=tmpacc
      print("New top (9->4): "+str(bestsofar))
      os.system('mv bestMod9.hdf overallBest9.hdf')
      open("bestParams9.txt","w+").write(str(param)+', acc:'+str(bestsofar))
