
"""
Importing the Neccessary Libraries for hw2
Winnow Algorithm

The WINNOW ALGORITHM IS IMPLEMENTED IN THIS CODE
Note: NAME for Perceptron and winnow are used interchangably for convenenience 
But The WINNOW algorithm is implemented here

The COMMAND LINE ARGUMENTS REQUIRED ARE :-  Number of training examples, Number of Epochs, Learning Rate, data_folder

example :- python winnow.py 1000 20 2 data

"""

import struct
import numpy as np
import time
import sys 
import csv
import gzip
#import matplotlib.pyplot as plt
# Remove above comment, if you want to generate the Plots required for the learning curves

#start = time.time()

"""
Reading the Data Directly from the compressed .gz files

The input Folder name is passed as final arugment from the command line - filename
"""


def read_data(filename,File):
    with gzip.open(filename+'/'+File) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
"""
The Confusion Matrix, function , which is used for computing the F1 Scores
"""
def CM(Y_pred,Y_true):
    Con_Mat=np.zeros((11,11))
    TP=np.zeros(11)
    FP=np.zeros(11)
    FN=np.zeros(11)
    F=np.zeros(11)
    # Updating the values in the confusion matrix for multi-class classification
    for i in range(0,len(Y_pred)):
        Con_Mat[int(Y_true[i])][int(Y_pred[i])]=Con_Mat[int(Y_true[i])][int(Y_pred[i])]+1
        
    for i in range(0,11):
        for j in range(0,11):
            if(i==j):
                TP[i]=Con_Mat[i][j] # True Positive Count for each label
            else:
                FN[i]=FN[i]+Con_Mat[i][j]  # False Negative Count for each label
                FP[i]=FP[i]+Con_Mat[j][i]  # False Positive Count for each label
        if(TP[i]==0):
            F[i]=0
        else:
            F[i]=2*TP[i]/float(2*TP[i]+FP[i]+FN[i])  # F1 Score computation
    
    F1_Score=float(np.sum(F))/(len(np.unique(Y_true))) # MACRO F1 Score
    Accuracy=float(np.sum(TP))/(len(Y_pred))           # Accuracy computation
    
    return Accuracy,F1_Score

"""
Making Predictions for the digit type based on the argmax value of W^T.X 
Implemented using the Winnow Algorithm
"""

def predict(w,x):
    
    b=np.matmul(w,x[:-1])  # Finding value of W^T.X, for all 10 perceptrons
    y_p=np.argmax(b)      # Finding the best perceptron, with highest score or value
    
    return y_p             # Returning the class with the highest score


"""
Testing The Perceptron Algorithm
"""

def test_perceptron(test_data,test_label,w):
    
    test_data=np.c_[test_data,test_label]
    
    Y_tr=[]
    Y_pr=[]
    
    for i in range(0,len(test_data)):
            y_p=predict(w,test_data[i])  # Computing the predicted class or label
            y_t=test_data[i][-1]         # Actual True class label
            Y_tr.append(y_t)
            Y_pr.append(y_p)
    
    Test_Accuracy,Test_F1Score=CM(Y_pr,Y_tr)     # Computing Accuracy and F1 Score                   
    #print("Test: F1 Score: %f , Accuracy: %f" %(Test_F1Score,Test_Accuracy))
    
    return Test_F1Score,Test_Accuracy     # Returning Accuracy, F1 Score values
    

"""
Training The Perceptron Algorithm
Computing the weight vectors for the 10 perceptrons
"""

def train_perceptron(train_data,train_label,N_train,N_epoch,N_learn_rate):
    
    train_data=train_data[0:N_train]       # Train data, based on number of training examples
    train_label=train_label[0:N_train]
    
    train_data=np.c_[train_data,train_label]

    """
    Choose which type of initialization of weights you want, either 1) or 2) and comment out the other line
    1 - w=np.random.rand(10,785) -1 : Random Initialization of weights
    2 - w=np.ones((10,785))  : ONE initialization of weights
    Default : 1 initialization of weights, since it gives higher test F1 and train F1 Score
    Comment the other line, which you do not want to use for initialization of weights.
    """
    
    w=np.random.rand(10,785)     # Weights for each 10 perceptrons with random initialization between (0,1)
    w=np.ones((10,785))     # Weights for each 10 perceptrons with zero initialization
    
    a=np.zeros((10,785))      # Average Winnow Algorithm - PERFORMS BETTER THAN NORMAL WINNOW
    
    Y_true=[]
    Y_pred=[]
    
    for k in range(0,N_epoch):      # Running the Perceptron algorithm across epochs
        np.random.shuffle(train_data)  # Shuffling the training data each epoch
        for i in range(0,N_train):       # For every training instance
            y_p=predict(w,train_data[i])  # Predicting the class label or digit
            y_t=train_data[i][-1]         # True class label

            if(y_p!=int(y_t)):            # Updating weights, based on the comparison of true and predicted labels
                c=np.where(train_data[i][:-1]==1)   # Updating weights only for instance features with value = 1
                w[int(y_t),c]=w[int(y_t),c]*(N_learn_rate)   # Incrementing Weight Update positive
                
            array=np.matmul(w,train_data[i][:-1])   # Computing values of all the perceptrons
            #threshold=np.max(array)/(N_learn_rate)  # Computing the threshold, to decrement weights in the winnow algorithm
            threshold=784
            for j in range(0,len(array)):           # Threshold is computed dynamically, like regularization term hyperparameter
                if((j!=int(y_t))and(array[j]>=threshold)):  # Decrementing the weights of remaining perceptrons greater than threshold
                    c=np.where(train_data[i][:-1]==1)       # Updating weights only for instance features with value = 1
                    w[j,c]=w[j,c]/(N_learn_rate)   # Decrementing Weight Update negative              
                                        # ONE VS ALL Approach
            a=a+w          # Updating the average weight vectors for 10 perceptrons for average winnow
            Y_true.append(y_t)
            Y_pred.append(y_p)
                                     
    
    Train_Accuracy,Train_F1Score=CM(Y_pred,Y_true)          # Computing Accuracy and F1 Score                   
    #print("Training: F1 Score: %f , Accuracy: %f" %(Train_F1Score,Train_Accuracy))
    
    # return w, Train_F1Score,Train_Accuracy  # Vanilla implementation of Winnow Algorithm
    return w,Train_F1Score,Train_Accuracy   # Returning the 10 perceptron weight vectors, Accuracy, F1 Score values

    #return a,Train_F1Score,Train_Accuracy   # Average WINNOW Algorithms, gives higher TEST F1 Score

"""
Effect of Number of Epoch in Learning
Generating the learning curves
"""

def number_epoch():
    
    N_ep=np.zeros(19)
    F1_tr=np.zeros(19)
    F1_te=np.zeros(19)
    Acc_tr=np.zeros(19)
    Acc_te=np.zeros(19)
    
    for i in range(0,19):   # Plotting the F1 Score, Accuracy Learning curves vs Number of epochs
        N_epoch=10+i*5      # Varying the number of epochs
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
Generating the learning curves
"""

def training_set_size():
    
    N_tr=np.zeros(39)
    F1_tr=np.zeros(39)
    F1_te=np.zeros(39)
    Acc_tr=np.zeros(39)
    Acc_te=np.zeros(39)
    
    for i in range(0,39):    # Plotting the F1 Score, Accuracy Learning curves vs Training Example Size
        N_train=500+(i*250)  # Varying the Number of training examples size  
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
    
"""
Effect of Learning Rate 
Generating the learning curves
"""
        
def learn_rate():    
        
    N_lr=np.zeros(4)
    F1_tr=np.zeros(4)
    F1_te=np.zeros(4)
    Acc_tr=np.zeros(4)
    Acc_te=np.zeros(4)
    
    for i in range(0,4):    # Plotting the F1 Score, Accuracy Learning curves vs Learning Rate
        N_learn_rate=(i+2)    # Varying the Learning Rate
        N_lr[i]=N_learn_rate
        print("The Number of training examples is :- %d ." % (N_train))
        print("The Number of epochs is :- %d ." % (N_epoch))
        print("The Learning Rate :- %f ." % (N_learn_rate))
        w,F1_tr[i],Acc_tr[i]=train_perceptron(train_data,train_label,N_train,N_epoch,N_learn_rate)
        F1_te[i],Acc_te[i]=test_perceptron(test_data,test_label,w)
        
    
    plt.figure(5)
    plt.plot(N_lr,Acc_tr,'bo-',label = "Training Accuracy Score")
    plt.figure(6)
    plt.plot(N_lr,F1_tr,'bo-', label = "Training F1 Score")
    
    plt.figure(5)
    plt.plot(N_lr,Acc_te,'ro-', label = "Test Accuracy Score")
    plt.figure(6)
    plt.plot(N_lr,F1_te,'ro-', label = "Test F1 Score")
    
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

train_data=train_data[0:10000]   # Taking the first 10000 training examples
train_label=train_label[0:10000]

train_data=train_data/255.0        # Dividing by the gray scale max threshold value
train_data=(train_data>=0.5)       # Converting to binary featrure values
train_data=train_data.astype('int')

test_data=test_data/255.0         # Dividing by the gray scale max threshold value
test_data=(test_data>=0.5)        # Converting to binary featrure values
test_data=test_data.astype('int')

train_data=train_data.reshape((10000,784))    # Reshaping 28x28 image vector to 784x1 feature input
test_data=test_data.reshape((10000,784))      # Reshaping 28x28 image vector to 784x1 feature input

train_data=np.c_[train_data,np.ones((10000,1))]   # Adding the bias term , with feature value as 1 to every instance
test_data=np.c_[test_data,np.ones((10000,1))]     # Adding the bias term , with feature value as 1 to every instance

"""
Functions for Plotting the Learning Curves

Remove comments for the below lines of code to generate the learning curves

N_train=10000
N_epoch=50
N_learn_rate=2

training_set_size()

N_train=10000
N_epoch=50
N_learn_rate=2

number_epoch()

N_train=10000
N_epoch=50
N_learn_rate=2

learn_rate()
"""
                      # The hyperparameters used for the perceptron algorithm
N_train=int(arg[1])
N_epoch=int(arg[2])
N_learn_rate=float(arg[3]) 
                              # Training the Perceptron Algorithm for computing the weight vectors
w,Train_F1Score,Train_Accuracy=train_perceptron(train_data,train_label,N_train,N_epoch,N_learn_rate)        
                              # Testing the Perceptron Algorithm using the test data
Test_F1Score,Test_Accuracy=test_perceptron(test_data,test_label,w)   

print("Training F1 Score: %f " %(Train_F1Score))
#print("Training: F1 Score: %f , Accuracy: %f" %(Train_F1Score,Train_Accuracy))  

print("Test F1 Score: %f " %(Test_F1Score))
#print("Test: F1 Score: %f , Accuracy: %f" %(Test_F1Score,Test_Accuracy))   
    

# Timing Metrics
end = time.time()
#print("The time taken for the algorithm computation is :- %f seconds." % (end-start))