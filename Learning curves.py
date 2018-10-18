
"""
Importing the Neccessary Libraries
"""

import struct
import numpy as np
import time
import sys 
import csv
import gzip
import matplotlib.pyplot as plt


start = time.time()

"""
Reading the Data from the compressed .gz files

The input Folder name is passed as final arugment from the command line - filename

"""


def read_data(filename,File):
    with gzip.open(filename+'/'+File) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def CM(Y_pred,Y_true):
    Con_Mat=np.zeros((11,11))
    TP=np.zeros(11)
    FP=np.zeros(11)
    FN=np.zeros(11)
    F=np.zeros(11)
    
    for i in range(0,len(Y_pred)):
        Con_Mat[int(Y_true[i])][int(Y_pred[i])]=Con_Mat[int(Y_true[i])][int(Y_pred[i])]+1
        
    for i in range(0,11):
        for j in range(0,11):
            if(i==j):
                TP[i]=Con_Mat[i][j]
            else:
                FN[i]=FN[i]+Con_Mat[i][j]
                FP[i]=FP[i]+Con_Mat[j][i]
        if(TP[i]==0):
            F[i]=0
        else:
            F[i]=2*TP[i]/float(2*TP[i]+FP[i]+FN[i])
    
    F1_Score=float(np.sum(F))/(len(np.unique(Y_true))) 
    Accuracy=float(np.sum(TP))/(len(Y_pred))
    
    return Accuracy,F1_Score

"""
Making Predictions for the digit type based on the argmax value of W^T.X 
"""

def predict(w,x):
    
    b=np.matmul(w,x[:-1])
    y_p=np.argmax(b)
    
    return y_p


"""
Testing The Perceptron Algorithm
"""

def test_perceptron(test_data,test_label,w):
    
    test_data=np.c_[test_data,test_label]
    
    Y_tr=[]
    Y_pr=[]
    
    for i in range(0,len(test_data)):
            y_p=predict(w,test_data[i])
            y_t=test_data[i][-1]
            Y_tr.append(y_t)
            Y_pr.append(y_p)
    
    Test_Accuracy,Test_F1Score=CM(Y_pr,Y_tr)                        
    print("Test: F1 Score: %f , Accuracy: %f" %(Test_F1Score,Test_Accuracy))
    
    return Test_F1Score,Test_Accuracy
    

"""
Training The Perceptron Algorithm
"""

def train_perceptron(train_data,train_label,N_train,N_epoch,N_learn_rate):
    
    train_data=train_data[0:N_train]
    train_label=train_label[0:N_train]
    
    train_data=np.c_[train_data,train_label]
    
    w=2*np.random.rand(10,785)-1
    w=np.zeros((10,785))
    a=np.zeros((10,785))     # The average weight vectors
    
    Y_true=[]
    Y_pred=[]
    
    for k in range(0,N_epoch):      # Running the Perceptron algorithm across epochs
        np.random.shuffle(train_data)  # Shuffling the training data each epoch
        for i in range(0,N_train):       # For every training instance
            y_p=predict(w,train_data[i])  # Predicting the class label or digit
            y_t=train_data[i][-1]         # True class label
            if(y_p!=int(y_t)):            # Updating weights, based on the comparison of true and predicted labels
                w[int(y_t)]=w[int(y_t)]+((N_learn_rate)*(1))*train_data[i][:-1]   # Incrementing Weight Update positive
                
            array=np.matmul(w,train_data[i][:-1])   # Computing values of all the perceptrons' classes
            for j in range(0,len(array)):
                    if((j!=int(y_t))and(array[j]>=0)):  # Decrementing the weights of remaining perceptrons greater than 0 value
                        w[j]=w[j]+((N_learn_rate)*(-1))*train_data[i][:-1]   # Decrementing Weight Update negative         
                        # ONE VS ALL APPROACH
            Y_true.append(y_t)
            Y_pred.append(y_p)
            a=a+w                         
    
    Train_Accuracy,Train_F1Score=CM(Y_pred,Y_true)                        
    print("Training: F1 Score: %f , Accuracy: %f" %(Train_F1Score,Train_Accuracy))
    
    
    return a,Train_F1Score,Train_Accuracy

"""
Effect of Number of Epoch in Learning
"""

def number_epoch():
    
    N_ep=np.zeros(19)
    F1_tr=np.zeros(19)
    F1_te=np.zeros(19)
    Acc_tr=np.zeros(19)
    Acc_te=np.zeros(19)
    
    for i in range(0,19):
        N_epoch=10+i*5    
        N_ep[i]=N_epoch
        print("The Number of training examples is :- %d ." % (N_train))
        print("The Number of epochs is :- %d ." % (N_epoch))
        print("The Learning Rate :- %f ." % (N_learn_rate))
        w,F1_tr[i],Acc_tr[i]=train_perceptron(train_data,train_label,N_train,N_epoch,N_learn_rate)
        F1_te[i],Acc_te[i]=test_perceptron(test_data,test_label,w)
        
    
    plt.figure(1)
    plt.plot(N_ep,Acc_tr, label = "Training Accuracy Score")
    plt.figure(2)
    plt.plot(N_ep,F1_tr, label = "Training F1 Score")
    
    plt.figure(1)
    plt.plot(N_ep,Acc_te, label = "Test Accuracy Score")
    plt.figure(2)
    plt.plot(N_ep,F1_te, label = "Test F1 Score")
    
    plt.figure(1)
    plt.xlabel('Number of Epochs') 
    # naming the y axis 
    plt.ylabel('Accuracy') 
    # giving a title to my graph 
    plt.title('Accuracy vs Epochs')   
    # show a legend on the plot 
    plt.legend()   
    # function to show the plot 
    plt.savefig('Accuracy_Epoch.png')
    
    plt.figure(2)
    plt.xlabel('Number of Epochs') 
    # naming the y axis 
    plt.ylabel('F1 Score') 
    # giving a title to my graph 
    plt.title('F1 Scores vs Epochs')   
    # show a legend on the plot 
    plt.legend()   
    # function to show the plot 
    plt.savefig('F1_Score_Epoch.png') 

"""
Effect of Size of Training Set in Learning
"""

def training_set_size():
    
    N_tr=np.zeros(39)
    F1_tr=np.zeros(39)
    F1_te=np.zeros(39)
    Acc_tr=np.zeros(39)
    Acc_te=np.zeros(39)
    
    for i in range(0,39):
        N_train=500+(i*250)    
        N_tr[i]=N_train
        print("The Number of training examples is :- %d ." % (N_train))
        print("The Number of epochs is :- %d ." % (N_epoch))
        print("The Learning Rate :- %f ." % (N_learn_rate))
        w,F1_tr[i],Acc_tr[i]=train_perceptron(train_data,train_label,N_train,N_epoch,N_learn_rate)
        F1_te[i],Acc_te[i]=test_perceptron(test_data,test_label,w)
        
    
    plt.figure(3)
    plt.plot(N_tr,Acc_tr, label = "Training Accuracy Score")
    plt.figure(4)
    plt.plot(N_tr,F1_tr, label = "Training F1 Score")
    
    plt.figure(3)
    plt.plot(N_tr,Acc_te, label = "Test Accuracy Score")
    plt.figure(4)
    plt.plot(N_tr,F1_te, label = "Test F1 Score")
    
    plt.figure(3)
    plt.xlabel('Number of Training Examples') 
    # naming the y axis 
    plt.ylabel('Accuracy') 
    # giving a title to my graph 
    plt.title('Accuracy vs Number of Training Examples')   
    # show a legend on the plot 
    plt.legend()   
    # function to show the plot 
    plt.savefig('Accuracy_Trainsize.png')
    
    plt.figure(4)
    plt.xlabel('Number of Training Examples') 
    # naming the y axis 
    plt.ylabel('F1 Score') 
    # giving a title to my graph 
    plt.title('F1 Scores vs Number of Training Examples')   
    # show a legend on the plot 
    plt.legend()   
    # function to show the plot 
    plt.savefig('F1_Score_Trainsize.png')
    
        
def learn_rate():    
        
    N_lr=np.zeros(4)
    F1_tr=np.zeros(4)
    F1_te=np.zeros(4)
    Acc_tr=np.zeros(4)
    Acc_te=np.zeros(4)
    
    for i in range(0,4):
        N_learn_rate=0.00001*(10**(i+1))  
        N_lr[i]=N_learn_rate
        print("The Number of training examples is :- %d ." % (N_train))
        print("The Number of epochs is :- %d ." % (N_epoch))
        print("The Learning Rate :- %f ." % (N_learn_rate))
        w,F1_tr[i],Acc_tr[i]=train_perceptron(train_data,train_label,N_train,N_epoch,N_learn_rate)
        F1_te[i],Acc_te[i]=test_perceptron(test_data,test_label,w)
        
    
    plt.figure(5)
    plt.semilogx(N_lr,Acc_tr,'bo-',label = "Training Accuracy Score")
    plt.figure(6)
    plt.semilogx(N_lr,F1_tr,'bo-', label = "Training F1 Score")
    
    plt.figure(5)
    plt.semilogx(N_lr,Acc_te,'ro-', label = "Test Accuracy Score")
    plt.figure(6)
    plt.semilogx(N_lr,F1_te,'ro-', label = "Test F1 Score")
    
    plt.figure(5)
    plt.xlabel('Learning Rate') 
    # naming the y axis 
    plt.ylabel('Accuracy') 
    # giving a title to my graph 
    plt.title('Accuracy vs Learning Rate')   
    # show a legend on the plot 
    plt.legend()   
    # function to show the plot 
    plt.savefig('Accuracy_learn.png')
    
    plt.figure(6)
    plt.xlabel('Learning Rate') 
    # naming the y axis 
    plt.ylabel('F1 Score') 
    # giving a title to my graph 
    plt.title('F1 Scores vs Learning Rate')   
    # show a legend on the plot 
    plt.legend()   
    # function to show the plot 
    plt.savefig('F1_Score_learn.png')
    

"""
Reading the Hyperparameters for the Perceptron from the command line
"""

arg=sys.argv
N_train=int(arg[1])
N_epoch=int(arg[2])
N_learn_rate=float(arg[3])
filename=arg[4]

test_data=read_data(filename,"t10k-images-idx3-ubyte.gz")
test_label=read_data(filename,"t10k-labels-idx1-ubyte.gz")
train_data=read_data(filename,"train-images-idx3-ubyte.gz")
train_label=read_data(filename,"train-labels-idx1-ubyte.gz")

"""
Pre-Processing the Data Set
"""

train_data=train_data[0:10000]
train_label=train_label[0:10000]

train_data=train_data/255.0
train_data=(train_data>=0.5)
train_data=train_data.astype('int')

test_data=test_data/255.0
test_data=(test_data>=0.5)
test_data=test_data.astype('int')

train_data=train_data.reshape((10000,784))
test_data=test_data.reshape((10000,784))

train_data=np.c_[train_data,np.ones((10000,1))]
test_data=np.c_[test_data,np.ones((10000,1))]


N_train=10000
N_epoch=50
N_learn_rate=0.001

training_set_size()

N_train=10000
N_epoch=50
N_learn_rate=0.001

number_epoch()

N_train=10000
N_epoch=50
N_learn_rate=0.001

learn_rate()
 
N_train=int(arg[1])
N_epoch=int(arg[2])
N_learn_rate=float(arg[3])
           
w,Train_F1Score,Train_Accuracy=train_perceptron(train_data,train_label,N_train,N_epoch,N_learn_rate)        
           
Test_F1Score,Test_Accuracy=test_perceptron(test_data,test_label,w)   

print("Training F1 Score: %f " %(Train_F1Score))
#print("Training: F1 Score: %f , Accuracy: %f" %(Train_F1Score,Train_Accuracy))  

print("Test F1 Score: %f " %(Test_F1Score))
#print("Test: F1 Score: %f , Accuracy: %f" %(Test_F1Score,Test_Accuracy))          

# Timing Metrics
end = time.time()
print("The time taken for the algorithm computation is :- %f seconds." % (end-start))