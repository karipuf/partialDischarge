from scipy.io import loadmat
from scipy.signal import periodogram
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from keras.models import Sequential as sq
from keras.layers import Convolution1D,MaxPooling1D,Flatten,Dense,Dropout,AveragePooling1D
import pandas as pd
import numpy as np
import re,os,pdb,keras
import sklearn
from matplotlib import pyplot as pl

print('haaa')

def Folder2Matrix(path='.'):
    '''
    Load in all sample*.mat files from a folder and return
    as an array suitable for use by 1d convolutional network
    '''

    matFiles=[os.path.normpath(path+os.path.sep+tmp) for tmp in os.listdir(path) if re.compile(".+\.mat$").match(tmp)]
    timeseries=np.concatenate([np.array(loadmat(tmp)['X']) for tmp in matFiles],axis=1)
    return np.array([timeseries]).T

def CreateCNN(classifier,nFilters=150,lenFilters=30):
    ''' 
    Return conv. NN
    '''

    cnn=sq()
    cnn.add(Convolution1D(nFilters,lenFilters,input_shape=(2500,1),strides=1,activation='relu'))
    cnn.add(MaxPooling1D(50,25))
    cnn.add(Dropout(.5))
    cnn.add(Convolution1D(150,20,strides=1,activation="relu"))
    cnn.add(MaxPooling1D(10,5))
    cnn.add(Dropout(.5))

    cnn.add(Flatten())
    cnn.add(Dense(50,activation='relu'))
    cnn.add(Dropout(.35))
    cnn.add(Dense(2,activation="softmax"))
    adam=keras.optimizers.Adam(lr=0.0008,decay=1e-6)
    cnn.compile(loss='categorical_crossentropy',optimizer=adam)
    
    return cnn

def TensorSplit(x,y,test_size=0.25,random_state=0):
    trainVec=np.array([tmp>testSplit for tmp in np.random.rand(len(x),)])
    testVec=~trainVec
    return (x[trainVec],x[testVec],y[trainVec],y[testVec])

def CNNScore(model,testx,testy):
    return (testy[model.predict(testx)>0.5].sum()/float(len(testy)))

colnames=['Cold (CV)','Hot (CV)','Cold->Hot','Hot->Cold']
res=pd.DataFrame(columns=colnames)
aucres=pd.DataFrame(columns=colnames)
nres=pd.DataFrame(columns=['Cold->Hot (DT)','Hot->Cold (DT)','Cold->Hot (LDA)','Hot->Cold (LDA)'])

# To be run from inside the "Data" folder

rs=2
c=1000
#num_splits=5
num_splits=1
testSplit=0.3
#num_splits=10
#testSplit=0.1
classifiers=["'CNN1'"]
modelFile='bestmodel.hdf5'
batchSize=64

for classifier in classifiers:

    # Experimental Parameters
    exec('chosenClassifier='+classifier) 

    ############################################
    # First, work only with the low-temp data
    #
    # Surface -> class 1
    # Void (internal) -> class 0
    #
    surf=Folder2Matrix('surface/')
    void=Folder2Matrix('void/')
    nsurf=len(surf)
    nvoid=len(void)

    # Generating data
    X=np.concatenate((surf,void))
    Y=np.concatenate((np.tile([1,0],(nsurf,1)),np.tile([0,1],(nvoid,1))))

    # Cross validating cold data
    scores=[]
    aucs=[]
    for rs in range(num_splits):
        print("(Cold) Cross validation round #"+str(rs+1)+"/"+str(num_splits))
        #trainx,testx,trainy,testy=cross_validation.train_test_split(X,Y,test_size=testSplit,random_state=rs);
        trainx,testx,trainy,testy=TensorSplit(X,Y,test_size=testSplit,random_state=rs);
        
        rfModel=CreateCNN(chosenClassifier)

        earlyStop=keras.callbacks.EarlyStopping(patience=3)
        saveBest=keras.callbacks.ModelCheckpoint(modelFile,save_best_only=True)
        rfModel.fit(trainx,trainy,epochs=50,validation_split=.1,callbacks=[earlyStop,saveBest],batch_size=batchSize);
        rfModel=keras.models.load_model(modelFile)
        scores.append(CNNScore(rfModel,testx,testy))
        aucs.append(roc_auc_score(testy,[[tmp] for tmp in rfModel.predict(testx)[:,1]]))

    print("Mean accuracy is: "+'{0:.2f}'.format(np.mean(scores)*100)+"%")
    print()

    cv1=np.mean(scores)*100;
    auccv1=np.mean(aucs)

    ##################
    # Hi-temp data

    surfHot=Folder2Matrix('surface/Heated/')
    voidHot=Folder2Matrix('void/Heated 10mv standard/')
    nsurfHot=len(surfHot)
    nvoidHot=len(voidHot)

    # Generating data
    HX=np.concatenate((surfHot,voidHot))
    HY=np.concatenate((np.tile([1,0],(nsurfHot,1)),np.tile([0,1],(nvoidHot,1))))
 
    # Cross validating hot data
    scores=[]
    aucs=[]
    for rs in range(num_splits):
        print("(Hot) Cross validation round #"+str(rs+1)+"/"+str(num_splits))
        #trainx,testx,trainy,testy=cross_validation.train_test_split(HX,HY,test_size=testSplit,random_state=rs);
        trainx,testx,trainy,testy=TensorSplit(HX,HY,test_size=testSplit,random_state=rs);
        rfModel=CreateCNN(chosenClassifier)

        saveBest=keras.callbacks.ModelCheckpoint(modelFile,save_best_only=True)
        earlyStop=keras.callbacks.EarlyStopping(patience=3)
        rfModel.fit(trainx,trainy,epochs=50,validation_split=.1,callbacks=[earlyStop],batch_size=batchSize);        
        rfModel=keras.models.load_model(modelFile)
        scores.append(CNNScore(rfModel,testx,testy))
        aucs.append(roc_auc_score(testy,[[tmp] for tmp in rfModel.predict(testx)[:,1]]))

    print("Mean accuracy is: "+'{0:.2f}'.format(np.mean(scores)*100)+"%")
    print()

    cv2=np.mean(scores)*100
    auccv2=np.mean(aucs)
                    

    ###################################################
    # Cross Scores (hot model on cold and vice versa)

    coldModel=CreateCNN(chosenClassifier)

    saveBest=keras.callbacks.ModelCheckpoint(modelFile,save_best_only=True)
    earlyStop=keras.callbacks.EarlyStopping(patience=3)
    coldModel.fit(X,Y,epochs=50,validation_split=.1,callbacks=[earlyStop],batch_size=batchSize);
    coldModel=keras.models.load_model(modelFile)
    crossscore1=CNNScore(coldModel,HX,HY)*100
    crossauc1=roc_auc_score(HY,[[tmp] for tmp in coldModel.predict(HX)[:,1]])

    hotModel=CreateCNN(chosenClassifier)

    saveBest=keras.callbacks.ModelCheckpoint(modelFile,save_best_only=True)
    earlyStop=keras.callbacks.EarlyStopping(patience=3)
    hotModel.fit(HX,HY,epochs=50,validation_split=.1,callbacks=[earlyStop],batch_size=batchSize);
    hotModel=keras.models.load_model(modelFile)
    crossscore2=CNNScore(hotModel,X,Y)*100
    crossauc2=roc_auc_score(Y,[[tmp] for tmp in hotModel.predict(X)[:,1]])

    print("Accuracy with the hot data (trained on cold) is "+'{0:.2f}'.format(crossscore1)+"%")
    print("Accuracy with the cold data (trained on hot) is "+'{0:.2f}'.format(crossscore2)+"%")

    res.loc[classifier]=[cv1,cv2,crossscore1,crossscore2]
    aucres.loc[classifier]=[auccv1,auccv2,crossauc1,crossauc2]
    








    


#for count in range(4):
#    res['auc'+str(count+1)]=aucres.iloc[:,count]
#res=res[[res.columns[tmp] for tmp in [0,4,1,5,2,6,3,7]]]
    
######################################################
# Training using noisy data

for noiseLevel in []: # [0,0.1,0.2,0.3,0.4,0.5]:

    crossscore1=0;
    crossscore2=0;
    crossscore3=0;
    crossscore4=0

    for count in range(5):
    
        # Standard deviation of each feature
        st=np.tile(X.std(axis=0),(X.shape[0],1,1))
        hst=np.tile(HX.std(axis=0),(HX.shape[0],1,1))

        # Creating noisy samples
        np.random.seed(rs+count)
        nX=X+np.random.randn(*X.shape)*st*noiseLevel
        nHX=HX+np.random.randn(*HX.shape)*hst*noiseLevel

        coldModel=CreateCNN(chosenClassifier)
        hotModel=CreateCNN(chosenClassifier);

        earlyStop=keras.callbacks.EarlyStopping(patience=3)
        coldModel.fit(nX,Y,epochs=50,validation_split=.1,callbacks=[earlyStop],batch_size=batchSize)
        hotModel.fit(nHX,HY,epochs=50,validation_split=.1,callbacks=[earlyStop],batch_size=batchSize)
        crossscore1+=CNNScore(coldModel,HX,HY)*100
        crossscore2+=CNNScore(hotModel,X,Y)*100
       
    nres.loc[str(noiseLevel*100)+"%"]=[crossscore1/5,crossscore2/5,crossscore3/5,crossscore4/5]


