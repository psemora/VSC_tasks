import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import random

"""
VSC project - Adaptive ANN
Petr Šemora, 4pAIŘ/1, LS 2021
minimizing method: Nelder-Mead
ANN architecture for the Rastrigin and Schwefel fuctions:
number of neurons / activation
512 / relu
256 / relu
128 / relu
64  / relu
32  / relu
16  / relu
8   / relu
4   / relu
1   / linear
loss function: mean_squared_error
optimizer: Adam
metrics: mae
epochs: 500
"""

class MainWindow:
    def __init__(self):
        self.x_schwef = []
        self.yt_schwef = []
        self.opt_schew_x = np.array([])
        self.opt_schew_y = np.array([])
        self.opt_rastr_x = np.array([])
        self.opt_rastr_y = np.array([])

    def rastrigin(self, X, Y):
        return (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20

    def schwefel(self, X, Y):
        return 418.9829*2 - (X*np.sin(np.sqrt(np.absolute(X)))+Y*np.sin(np.sqrt(np.absolute(Y))))

    def input_data(self):
        v = np.linspace(-5.12, 5.12, 40)
        v2 = np.linspace(-500, 500, 40)

        self.X, self.Y = np.meshgrid(v, v) 
        self.X2, self.Y2 = np.meshgrid(v2, v2)
        
        self.x_rastr = np.random.rand(10000,2)*10.24-5.12 #<-5.12;5.12>  #rastrigin input
        self.x_schwef = np.random.rand(10000,2)*1000-500 #<-500;500>  #schwefel input

        self.yt_rastr = np.reshape(self.rastrigin(self.x_rastr[:,0],self.x_rastr[:,1]), (self.x_rastr.shape[0],1)) 
        self.yt_schwef = np.reshape(self.schwefel(self.x_schwef[:,0],self.x_schwef[:,1]), (self.x_schwef.shape[0],1))


    def train_rastr(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x_rastr, self.yt_rastr, test_size=0.3, random_state=0)
        #Rastrigin ANN architecture
        net = Sequential()
        net.add(Dense(512, input_dim=2, activation ='relu'))
        net.add(Dense(256, activation ='relu'))
        net.add(Dense(128, activation ='relu'))
        net.add(Dense(64, activation ='relu'))
        net.add(Dense(32, activation ='relu'))
        net.add(Dense(16, activation ='relu'))
        net.add(Dense(8, activation ='relu'))
        net.add(Dense(4, activation ='relu'))
        net.add(Dense(1, activation ='linear'))
        net.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])
        net.fit(x_train,y_train, epochs=500, batch_size=64, verbose = 0)
        test_loss, test_acc = net.evaluate(x_test, y_test)
        print('Test loss:', test_loss)
        net.save('rastrigin.model')

    def train_schwef(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x_schwef, self.yt_schwef, test_size=0.3, random_state=0)
        #Schwefel ANN Achitecture
        net = Sequential()
        net.add(Dense(512, input_dim=2, activation ='relu'))
        net.add(Dense(256, activation ='relu'))
        net.add(Dense(128, activation ='relu'))
        net.add(Dense(64, activation ='relu'))
        net.add(Dense(32, activation ='relu'))
        net.add(Dense(16, activation ='relu'))
        net.add(Dense(8, activation ='relu'))
        net.add(Dense(4, activation ='relu'))
        net.add(Dense(1, activation ='linear'))
        net.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])
        net.fit(x_train,y_train, epochs=500, batch_size=64, verbose = 0)
        test_loss, test_acc = net.evaluate(x_test, y_test)
        print('Test loss:', test_loss)
        net.save('schwefel.model')

    def predict_rastr(self):
        self.net_rastr = load_model('trained_rastrigin.model')
        self.Z = np.zeros((40, 40))
        for i in range(40):
            for j in range(40):
                self.Z[i, j] = self.net_rastr.predict(np.array([[self.X[i,j], self.Y[i,j]]]))
        return 

    def predict_schwef(self):
        self.net_schwef = load_model('trained_schwefel.model')
        self.Z2 = np.zeros((40, 40))
        for i in range(40):
            for j in range(40):
                self.Z2[i, j] = self.net_schwef.predict(np.array([[self.X2[i,j], self.Y2[i,j]]]))
        return
    
    def pred_rastr(self, x):   
        return self.net_rastr.predict(np.array([[x[0],x[1]]]))
    
    def pred_schwef(self, x): 
        return self.net_schwef.predict(np.array([[x[0],x[1]]]))

    def search_rastr(self):
        r_min, r_max = -5.12, 5.12
        init_value = np.random.rand(2) * (r_max - r_min) + r_min 
        result = minimize(self.pred_rastr, init_value, method='nelder-mead', options={'adaptive':True})
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev']) 
        self.solution = result['x']
        self.evaluation = self.pred_rastr(self.solution)
        print('Minimum: (%s) = %.5f' % (self.solution, self.evaluation))
        self.opt_rastr_x = np.append(self.opt_rastr_x, [self.solution])
        self.opt_rastr_y = np.append(self.opt_rastr_y, [self.evaluation])
    
    def search_schwef(self):
        r_min, r_max = -500, 500
        init_value = np.random.rand(2) * (r_max - r_min) + r_min 
        result = minimize(self.pred_schwef, init_value, method='nelder-mead', options={'adaptive':True})
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev']) 
        self.solution = result['x']
        self.evaluation = self.pred_schwef(self.solution)
        print('Solution: f(%s) = %.5f' % (self.solution, self.evaluation))
        self.opt_schew_x = np.append(self.opt_schew_x, [self.solution])
        self.opt_schew_y = np.append(self.opt_schew_y, [self.evaluation])


    def show_rastrigin(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #Z_orig = self.rastrigin(self.X, self.Y) #rastrigin original
        ax.plot_surface(self.X, self.Y, self.Z)  
        ax.scatter3D(self.solution[0], self.solution[1], self.evaluation, color = "red")  #minimum 
        #ax.plot_surface(self.X, self.Y, Z_orig-self.Z) #rastrigin mistake
        #ax.plot(self.x_rastr[:,0], self.x_rastr[:,1], np.ones(self.x_rastr.shape[0])*10,'b*')  #input points
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        ax.set_title('Rastrigin function')
        plt.xlim(-5.12, 5.12)
        plt.ylim(-5.12, 5.12)
        ax.set_zlim(0,90)
        ax.grid()
        plt.show()

    def show_schwefel(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #Z_orig = self.schwefel(self.X2, self.Y2) #schwefel original
        ax.plot_surface(self.X2, self.Y2, self.Z2) 
        ax.scatter3D(self.solution[0], self.solution[1], self.evaluation, color = "red")  #minimum 
        #ax.plot_surface(self.X2, self.Y2, Z_orig-self.Z2) #schwefel mistake
        #ax.plot(self.x_schwef[:,0], self.x_schwef[:,1], np.ones(self.x_schwef.shape[0])*10,'b*')  #input points
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        ax.set_title('Schwefel function')
        plt.xlim(-500, 500)
        plt.ylim(-500, 500)
        ax.set_zlim(0,1600)
        ax.grid()
        plt.show()
    
    def iteration_rastr(self):
        for i in range(4):
            print("\nIteration number: ", i+1)
            #self.train_rastr() #comment this line to use pretrained network "trained_rastrigin.model"
            self.predict_rastr()
            self.search_rastr()
            self.x_rastr = np.concatenate((self.x_rastr, [self.solution]))
            self.yt_rastr = np.append(self.yt_rastr, [self.evaluation])
        self.show_rastrigin()

    def iteration_schwef(self):
        for i in range(4):
            print("\nIteration number: ", i+1)
            #self.train_schwef() #comment this line to use pretrained network "trained_schwefel.model"
            self.predict_schwef()   
            self.search_schwef()
            self.x_schwef = np.concatenate((self.x_schwef, [self.solution]))
            self.yt_schwef = np.append(self.yt_schwef, [self.evaluation])
        self.show_schwefel()

if __name__ == "__main__":
    window = MainWindow()
    window.input_data()
    choice = input("Enter 1 for train the Rastrigin function, enter 2 for train the Schwefel function: ")
    if choice == "1":
        window.iteration_rastr()
    elif choice == "2":
        window.iteration_schwef()
    else:
        print("Wrong input !")