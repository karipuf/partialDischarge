from partialdischarge_funcs import *
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real,Integer

randSeed=10
testSize=.1
nCalls=30
resFile='skopt_result.pkl'

###################
# Temperature!

temp=CreateTemperatureData()

# Cold, first
Xcold,Xhot,Ycold,Yhot=[temp[tmp] for tmp in ['Xcold','Xhot','Ycold','Yhot']]
xtrain,xtest,ytrain,ytest=train_test_split(Xcold,Ycold,test_size=testSize,random_state=randSeed)


# This one seems to work well for the cold->hot case
# cnn=CreateCNN((100,150,150,150,200),(10,10,10,10,10),poolSizes=(3,3,3,3,3),lr=.0001,dropProb=.7)

# GP optmization
space=[Integer(3,6),Integer(5,15),Real(.00001,.001),Real(.5,.8)]
tf=open("skopt_tmp.txt","a+")
counter=iter(range(1000000))

def sc(tmp):
    tf.write("Scikit-opt Round #"+str(next(counter))+"\n")
    tf.flush()

    nFilts=(100,150,200,250,250,300)[:tmp[0]]
    lenFilts=[int(tmp[1])]*tmp[0]
    pools=[3]*tmp[0]
    cnn=CreateCNN(nFilts=nFilts,lenFilts=lenFilts,poolSizes=pools)
    
    #cnn=CreateCNN(nFilts,lenFilts,pools,lr=tmp[2],dropProb=tmp[3])
    
    
    cnn.fit(xtrain,ytrain,epochs=250,batch_size=128)
    return cnn.evaluate(Xhot,Yhot)[1]

res=gp_minimize(sc,space,n_calls=nCalls)
# Wrapping up
print(str(res))
pickle.dump([res['x'],res['fun']],open(resFile,"wb+"))
