from partialdischarge_funcs import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint,CSVLogger
from itertools import count
import numpy as np
import pylab as pl
import pandas as pd
import keras,random,sys

randSeed=10
testSize=.01

###################
# Temperature

temp=CreateTemperatureData()

# Cold, first
Xcold,Xhot,Ycold,Yhot=[temp[tmp] for tmp in ['Xcold','Xhot','Ycold','Yhot']]
xtrain,xtest,ytrain,ytest=train_test_split(Xcold,Ycold,test_size=testSize,random_state=randSeed)
xtrainhot,xtesthot,ytrainhot,ytesthot=train_test_split(Xhot,Yhot,test_size=testSize,random_state=randSeed)

################
# Cold->Hot
################

nParams=80
nfiltbases=np.random.choice(range(50,150),nParams)
filtlens=np.random.choice(range(6,12),nParams)
lrs=np.random.choice([0.01,.005,.001,.0005,.0001],nParams)
dropProbs=np.random.choice([.3,.4,.5,.6,.7,],nParams)

params=zip(nfiltbases,filtlens,lrs,dropProbs)
bestsofar=0

counter=count(1)
for param in params:
    
    print("(cold->hot) Running paramset #"+str(next(counter)))
    sys.stdout.flush()
    
    nfiltbase,filtlen,lr,dropProb=param
    nfilts=[int(nfiltbase*tmp) for tmp in [1,1.5,1.5,1.5,2]]   
    cnn=CreateCNN(nfilts,[int(filtlen)]*len(nfilts),poolSizes=(3,3,3,3,3),lr=lr,dropProb=dropProb,reg=.0)  
    callbacks=[ModelCheckpoint('bestMod.hdf',save_best_only=True,monitor='val_acc'),CSVLogger('trainres.csv')]

    cnn.fit(xtrain,ytrain,epochs=100,batch_size=64,callbacks=callbacks,validation_split=.05,verbose=0)
    
    cnnbest=keras.models.load_model('bestMod.hdf')
    tmpacc=np.sum(cnnbest.predict(Xhot).argmax(axis=1)==Yhot.argmax(axis=1))/len(Yhot)
    
    if tmpacc>bestsofar:
        bestsofar=tmpacc
        print("New top: "+str(bestsofar))
        cnnbest.save('overallBest.hdf')
        open("bestParams.txt","w+").write(str(param)+', acc:'+str(bestsofar))


# redoing the cold->hot ;-(
#sys.exit()

###############
# Hot->Cold
###############
        
nParams=80
nfiltbases=np.random.choice(range(50,150),nParams)
filtlens=np.random.choice(range(6,12),nParams)
lrs=np.random.choice([0.01,.005,.001,.0005,.0001],nParams)
dropProbs=np.random.choice([.3,.4,.5,.6,.7,],nParams)

params=zip(nfiltbases,filtlens,lrs,dropProbs)
bestsofar=0

counter=count(1)
for param in params:
    
    print("Running paramset #"+str(next(counter)))
    sys.stdout.flush()
    
    nfiltbase,filtlen,lr,dropProb=param
    nfilts=[int(nfiltbase*tmp) for tmp in [1,1.5,1.5,1.5,2]]   
    cnnhot=CreateCNN(nfilts,[int(filtlen)]*len(nfilts),poolSizes=(3,3,3,3,3),lr=lr,dropProb=dropProb,reg=.0)
    
    callbacks=[ModelCheckpoint('bestModHot.hdf',save_best_only=True,monitor='val_acc'),CSVLogger('trainreshot.csv')]
    cnnhot.fit(xtrainhot,ytrainhot,epochs=100,batch_size=64,callbacks=callbacks,validation_split=.05,verbose=0) # validation_data=(xtesthot,ytesthot)
    
    cnnhotbest=keras.models.load_model('bestModHot.hdf')
    tmpacc=np.sum(cnnhotbest.predict(Xcold).argmax(axis=1)==Ycold.argmax(axis=1))/len(Ycold)
    
    if tmpacc>bestsofar:
        bestsofar=tmpacc
        print("New top: "+str(bestsofar))
        cnnhotbest.save('overallBestHot.hdf')
        open("bestParamsHot.txt","w+").write(str(param)+', acc:'+str(bestsofar))

