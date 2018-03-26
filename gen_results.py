from partialdischarge_funcs import *
from sklearn.model_selection import train_test_split


randSeed=10
testSize=.1

###################
# Temperature!

temp=CreateTemperatureData()

# Cold, first
Xcold,Xhot,Ycold,Yhot=[temp[tmp] for tmp in ['Xcold','Xhot','Ycold','Yhot']]
xtrain,xtest,ytrain,ytest=train_test_split(Xcold,Ycold,test_size=testSize,random_state=randSeed)


cnn=CreateCnn()
