import pandas as pd
import seaborn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def sigmoid_derivative(z):
        return z * (1.0 - z)

class NeuralNetwork:
    def __init__(self, inSize, sl2,clsSize, lrt):

        self.iSz=inSize
        self.oSz=clsSize
        self.hSz=sl2
        self.weights1 = (np.random.rand(self.hSz,self.iSz+1)-0.5)/np.sqrt(self.iSz)
        self.weights2 = (np.random.rand(self.oSz,self.hSz+1)-0.5)/np.sqrt(self.hSz)

        self.output=0
        self.layer1=np.zeros(self.hSz)
        self.eta=lrt


    def feedforward(self, x):
        x_c=np.r_[1,x]
        self.layer1 = sigmoid(np.dot(self.weights1,x_c))
        layer1_c=np.r_[1,self.layer1]
        self.output = sigmoid(np.dot(self.weights2,layer1_c))


    def backprop(self,x, trg):

        sigma_3=(trg-self.output)  # outer layer error\n",
        sigma_3=np.reshape(sigma_3,(self.oSz,1))

        layer1_c=np.r_[1,self.layer1]    # hidden layer activations+bias\n",
        sigma_2=np.dot(self.weights2.T,sigma_3)
        tmp=sigmoid_derivative(layer1_c)
        tmp=np.reshape(tmp,( self.hSz+1,1))
        sigma_2=np.multiply(sigma_2,tmp)      # hidden layer error \n",
        delta2=sigma_3*layer1_c   # weights2 update\n",


        x_c=np.r_[1,x]      # input layer +bias\n",
        delta1=sigma_2[1:,]*x_c # weights1 update\n",

        return delta1,delta2

    def fit(self,X,y,iterNo):

        m=np.shape(X)[0]
        for i in range(iterNo):
            D1=np.zeros(np.shape(self.weights1))
            D2=np.zeros(np.shape(self.weights2))
            for j in range(m):
                self.feedforward(X[j])
                [delta1,delta2]= self.backprop(X[j],y[j])
                D1=D1+delta1
                D2=D2+delta2
            self.weights1= self.weights1+self.eta*(D1/m)
            self.weights2=self.weights2+self.eta*(D2/m)


    def predict(self,X):

        m=np.shape(X)[0]
        y=np.zeros((m, self.oSz))
        for i in range(m):
            self.feedforward(X[i])
            y[i] = self.output
        return y





iris = datasets.load_iris()


X = iris['data']
Y = iris['target']

order = range(np.shape(X)[0])
allocation = list(order)
np.random.shuffle(allocation)


X_shuffled = np.zeros((X.shape[0],X.shape[1]))
Y_shuffled = np.zeros((X.shape[0],), dtype=int)
for i in range(X.shape[0]):
    index = allocation[i]
    X_shuffled[i] = X[index]
    Y_shuffled[i] = Y[index]

#print(Y_shuffled)
#print(X_shuffled)

Y_shuffled_3 = np.zeros((X.shape[0],3), dtype=int)
for i in range(Y_shuffled.shape[0]):
    if Y_shuffled[i] == 0:
        Y_shuffled_3[i]= np.array([1,0,0])
    elif Y_shuffled[i] == 1:
        Y_shuffled_3[i]= np.array([0,1,0])
    elif Y_shuffled[i] == 2:
        Y_shuffled_3[i]= np.array([0,0,1])



training_data_x = X_shuffled[:100, :]
training_data_y = Y_shuffled_3[:100, :]

validation_data_x = X_shuffled[100:125, :]
validation_data_y = Y_shuffled_3[100:125,:]

testing_data_x = X_shuffled[125:150, :]
testing_data_y = Y_shuffled_3[125:150,:]

#testing_data_yy = np.zeros((X.shape[0],), dtype=int)

#print(testing_data_y.shape)



nnList = []
for i in range(10):
    nn = NeuralNetwork(4, 4, 3, 0.25)
    nnList.append(nn)




#print(training_data_x.T.shape)

#error = np.zeros((X.shape[0],3), dtype=int)
error_list = np.zeros(10)
i = 0
prediction_list = []
for nn in nnList:
    nn.fit(training_data_x,training_data_y,600)



for nn in nnList:
    predictions = nn.predict(validation_data_x)
    error = (np.subtract(validation_data_y,predictions))**2
    error_list[i] = (np.sum(error, dtype= np.float))
    i = i + 1



print(error_list)
index_min = np.argmin(error_list)

print(index_min)


selected_nn = nnList[index_min]

last_prediction = selected_nn.predict(testing_data_x)
last_pred = []
test_y = Y_shuffled[125:150]

for i in range(last_prediction.shape[0]):
    row =np.array(last_prediction[i])
    last_pred.append(np.argmax(row))

conf_matrix = []
for i in range(3): # initialize confusion matrix
    temp = []
    for j in range(3):
        temp.append(0)
    conf_matrix.append(temp)


hit = 0 #dogru tahmin sayisi

for i in range(len(last_pred)):
    conf_matrix[test_y[i]][last_pred[i]] += 1
    print(test_y[i])
    print(last_pred[i])
    if test_y[i] == last_pred[i]:
        hit = hit + 1




print('hit = %d' % hit)
accuracy = hit/testing_data_y.shape[0]
print('accuracy is %f ' % accuracy)


df_cm = pd.DataFrame(conf_matrix, index=[i for i in [0,1,2]],
                     columns=[i for i in [0,1,2]])
plt.figure(figsize = (10,7))
plt.title('confusion matrix')
seaborn.heatmap(df_cm, annot=True)
plt.show()

