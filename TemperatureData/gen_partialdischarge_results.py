from scipy.io import loadmat
from scipy.signal import periodogram
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import re,os,pdb
import sklearn
import pylab

def Folder2Matrix(path='.'):

    '''
    Load in all sample*.mat files from a folder and return as an array n x dim
    '''

    matFiles=[os.path.normpath(path+os.path.sep+tmp) for tmp in os.listdir(path) if re.compile(".+\.mat$").match(tmp)]
    timeseries=np.concatenate([np.array(loadmat(tmp)['X']) for tmp in matFiles],axis=1).transpose()
    return periodogram(timeseries)[1][:,:200]

def InitModel(chosenClassifier):
    
    if chosenClassifier==sklearn.svm.classes.SVC:
        retModel=chosenClassifier(random_state=rs,kernel='linear',C=c,probability=True)
    else:
        if (chosenClassifier==sklearn.lda.LDA) or (chosenClassifier==sklearn.qda.QDA):
            retModel=chosenClassifier()
        else:
            if (chosenClassifier==sklearn.ensemble.forest.RandomForestClassifier):
                retModel=chosenClassifier(random_state=rs,n_estimators=100)
            else:
                retModel=chosenClassifier(random_state=rs)

    #if chosenClassifier==sklearn.qda.QDA:
    #    retModel=chosenClassifier(reg_param=0.1)

    return retModel


colnames=['Cold (CV)','Hot (CV)','Cold->Hot','Hot->Cold']
res=pd.DataFrame(columns=colnames)
aucres=pd.DataFrame(columns=colnames)
nres=pd.DataFrame(columns=['Cold->Hot (DT)','Hot->Cold (DT)','Cold->Hot (LDA)','Hot->Cold (LDA)'])

# To be run from inside the "Data" folder

rs=2
c=1000
num_splits=5
testSplit=0.3
#num_splits=10
#testSplit=0.1
classifiers=['RandomForestClassifier','GradientBoostingClassifier','SVC','DecisionTreeClassifier','LDA']


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
    Y=np.concatenate((np.ones(nsurf),np.zeros(nvoid)))

    # Cross validating cold data
    scores=[]
    aucs=[]
    for rs in range(num_splits):
        print "(Cold) Cross validation round #"+str(rs+1)+"/"+str(num_splits)
        trainx,testx,trainy,testy=cross_validation.train_test_split(X,Y,test_size=testSplit,random_state=rs);
        
        rfModel=InitModel(chosenClassifier)
        rfModel.fit(trainx,trainy);
        scores.append(rfModel.score(testx,testy))
        aucs.append(roc_auc_score(testy,rfModel.predict_proba(testx)[:,1]))

    print "Mean accuracy is: "+'{0:.2f}'.format(np.mean(scores)*100)+"%"
    print

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
    HY=np.concatenate((np.ones(nsurfHot),np.zeros(nvoidHot)))

    # Cross validating hot data
    scores=[]
    aucs=[]
    for rs in range(num_splits):
        print "(Hot) Cross validation round #"+str(rs+1)+"/"+str(num_splits)
        trainx,testx,trainy,testy=cross_validation.train_test_split(HX,HY,test_size=testSplit,random_state=rs);

        rfModel=InitModel(chosenClassifier)
        rfModel.fit(trainx,trainy);        
        scores.append(rfModel.score(testx,testy))
        aucs.append(roc_auc_score(testy,rfModel.predict_proba(testx)[:,1]))

    print "Mean accuracy is: "+'{0:.2f}'.format(np.mean(scores)*100)+"%"
    print

    cv2=np.mean(scores)*100
    auccv2=np.mean(aucs)
                    

    ###################################################
    # Cross Scores (hot model on cold and vice versa)

    coldModel=InitModel(chosenClassifier)
    coldModel.fit(X,Y);
    crossscore1=coldModel.score(HX,HY)*100
    crossauc1=roc_auc_score(HY,coldModel.predict_proba(HX)[:,1])

    hotModel=InitModel(chosenClassifier)   
    hotModel.fit(HX,HY);
    crossscore2=hotModel.score(X,Y)*100
    crossauc2=roc_auc_score(Y,hotModel.predict_proba(X)[:,1])

    print "Accuracy with the hot data (trained on cold) is "+'{0:.2f}'.format(crossscore1)+"%"
    print "Accuracy with the cold data (trained on hot) is "+'{0:.2f}'.format(crossscore2)+"%"

    res.loc[classifier]=[cv1,cv2,crossscore1,crossscore2]
    aucres.loc[classifier]=[auccv1,auccv2,crossauc1,crossauc2]
    

#for count in range(4):
#    res['auc'+str(count+1)]=aucres.iloc[:,count]
#res=res[[res.columns[tmp] for tmp in [0,4,1,5,2,6,3,7]]]
    
######################################################
# Training using noisy data

    
for noiseLevel in [0,0.1,0.2,0.3,0.4,0.5]:

    crossscore1=0;
    crossscore2=0;
    crossscore3=0;
    crossscore4=0

    for count in range(5):
    
        # Standard deviation of each feature
        st=np.tile(X.std(axis=0),(X.shape[0],1))
        hst=np.tile(HX.std(axis=0),(HX.shape[0],1))

        # Creating noisy samples
        pylab.seed(rs+count)
        nX=X+pylab.randn(*X.shape)*st*noiseLevel
        nHX=HX+pylab.randn(*HX.shape)*hst*noiseLevel

        coldModel=InitModel(DecisionTreeClassifier)
        hotModel=InitModel(DecisionTreeClassifier);
        
        coldModellda=InitModel(LDA)
        hotModellda=InitModel(LDA)

        coldModel.fit(nX,Y)
        hotModel.fit(nHX,HY)
        coldModellda.fit(nX,Y)
        hotModellda.fit(nHX,HY)
        crossscore1+=coldModel.score(HX,HY)*100
        crossscore2+=hotModel.score(X,Y)*100
        crossscore3+=coldModellda.score(HX,HY)*100
        crossscore4+=hotModellda.score(X,Y)*100


    nres.loc[str(noiseLevel*100)+"%"]=[crossscore1/5,crossscore2/5,crossscore3/5,crossscore4/5]


