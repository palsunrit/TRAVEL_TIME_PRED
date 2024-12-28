#ImportingModules
import numpy as np
import matplotlib.pyplot as pl

#InitializeParamenters
def initialize(length):
    print("the length of weight vector is %d "%length)
    wt=np.random.randn(1,length)
    bias=0
    print()
    print("The weigth vector after initialization")
    print(wt)
    return wt,bias

#ForwardPropagation
def fp(X,wt,bias):
    z=np.dot(wt,X)+(bias)
    return z

#CostFunction
def loss(z,y):
    m=y.shape[0] 
    J=(1/(2*m))*np.sum(np.square(z-y))
    return J

#BackPropagation
def backPropagation(X,y,z):
    m=y.shape[0]   
    dz=(1/m )*(z-y)
    dw=np.dot(dz,X.T)
    db=np.sum(dz)
    return dw,db

#GradientDescentUpdate
def update(w,dw,b,db,l_rate):
    w=w-l_rate*dw
    b=b-l_rate*db
    return w,b

def linearRegression(X_train,Y_train,x_test,y_test,l_rate,epoch):
    length=X_train.shape[0] 
    w,b=initialize(length)
    
    m_train=Y_train.shape[0]
    m_val=y_test.shape[0]
    
    loss_train=[]
    costs_train=[]
    for e in range(0,epoch):
        z_train=fp(X_train,w,b)
        loss_train=loss(z_train,Y_train)
        
        dw,db=backPropagation(X_train,Y_train,z_train)
        w,b=update(w,dw,b,db,l_rate)
        if(e%500==0):
            costs_train.append(loss_train)

        MAE_train=(1/m_train)*np.sum(np.abs(z_train-Y_train))
        
        z_test=fp(x_test,w,b)  
        loss_test=loss(z_test,y_test)
        MAE_train=(1/m_val)*np.sum(np.abs(z_test-y_test))
        
        '''print("epoch "+str(e+1)+"/"+str(epoch))
        print("Training loss "+str(loss_train)+"|"+" Validation loss "+str(loss_test))
        print()'''
    
    y= fp(x_test,w,b)
    print("Mean squared error: %.4f" % np.mean((y- y_test)**2))
    print("the resuklt set is")
    print(y)
    print("The mean absolute error is %.4f"%MAE_train)
    
    #plotting
    pl.plot(costs_train) 
    pl.xlabel("Iterations")
    pl.ylabel("Training Cost")
    pl.title("Learning Rate "+str(l_rate))
    pl.show()
        
        
import pandas as pd
dataset=pd.read_csv('A:/Py_workspace/NeuralNetwork/TRAIN_SET.csv')

x= dataset.iloc[:,:-1]
y = dataset.iloc[:, 7]

ZoneDummy=pd.get_dummies(x['Zone'],prefix='Zone',drop_first=True)
x=pd.concat([x,ZoneDummy],axis=1)

DayDummy=pd.get_dummies(x['CodedDay'],prefix='Day',drop_first=True)
x=pd.concat([x,DayDummy],axis=1)

# Drop the unecessary coulmns
x=x.drop('Date',axis=1)
x=x.drop('Day',axis=1)
x=x.drop('Zone',axis=1)
x=x.drop('CodedDay',axis=1)


from sklearn import preprocessing
X=preprocessing.normalize(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=5)

X_train=X_train.T
y_train=np.array(y_train)
y_test=np.array(y_test)
#y_train=y_train.astype(int)
X_test=X_test.T


linearRegression(X_train,y_train,X_test,y_test,0.4,1000)
#print("ok")
#k=y_train.T.shape[0]
#print(k)

import statsmodels.api as sm
x=sm.add_constant(x)
lm_4=sm.OLS(y,x).fit()
print(lm_4.summary())

