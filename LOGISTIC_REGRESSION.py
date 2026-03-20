import numpy as np 
import pandas as pd  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
class Logistic_Regression(): 
    #declaraing number of iteration and learning rate
    def __init__(self,learning_rate,num_of_iteration):
        self.learning_rate = learning_rate
        self.num_of_iteration = num_of_iteration
    def fit(self,x,y):
        # m --> number of rows in the dataset
        # n --> number of columns in the dataset 
        self.m,self.n = x.shape
        self.x = x 
        self.y = y
        self.w = np.zeros(self.n)
        self.b = 0
        #implementing gradient descent for optimization
        for i in range(self.num_of_iteration):
            self.update_weights()


    def update_weights(self):
        y_hat = 1/(1+np.exp(-(self.x.dot(self.w)+self.b)))
        dw = (1/self.m)*np.dot(self.x.T,(y_hat - self.y))
        db = (1/self.m)*np.sum(y_hat-self.y)
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
        
    def predict(self,x):          
        y_predict =   1/(1+np.exp(-(x.dot(self.w)+self.b)))
        y_predict = np.where(y_predict>0.5,1,0)
        return y_predict
diabetes_data = pd.read_csv(r"C:\Users\samee\Downloads\diabetes (1).csv")
pd.set_option('display.max_columns',None)
diabetes_data.shape
diabetes_data.groupby('Outcome').mean()
features = diabetes_data.drop(columns= 'Outcome',axis = 1)
target = diabetes_data['Outcome']
scaler = StandardScaler()
scaler.fit(features)
standardised_data = scaler.transform(features)
x_train , x_test , y_train , y_test = train_test_split(standardised_data,target,test_size=0.2,random_state=2)
classifier = Logistic_Regression(learning_rate=0.01,num_of_iteration=1000)
classifier.fit(x_train,y_train)
x_train_prediction = classifier.predict(x_train)
train_data_accuracy = accuracy_score(y_train,x_train_prediction)
print("the accuracy of model is: ",train_data_accuracy)
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print("the accuracy of model is: ",test_data_accuracy)
#making a predictive model 
input_data = (2,197,70,45,543,30.5,0.158,53)
input_array = np.asarray(input_data)
input_data_reshaped = input_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
print(prediction)
 
