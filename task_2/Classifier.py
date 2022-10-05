from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import pandas 
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

"""
VSC Project, Petr Šemora, 4pAIŘ/1
ANN Binary Classifier with simple GUI
Inputs: - write in QLineEdits
        - click on the graph
        - load from CSV file data.csv
        X1: <-4,2>
        X2: <2,5>
Outputs: Prediction points outside or inside ellipse
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ANN Binary Classifier')
        self.setMinimumSize(QSize(400, 250))
        self.content = QWidget()
        layout = QGridLayout()

        self.a_input = QLineEdit('')
        self.b_input = QLineEdit('')

        layout.addWidget(QLabel('X1'), 0, 0)
        layout.addWidget(QLabel('X2'), 1, 0)

        layout.addWidget(self.a_input, 0, 1)
        layout.addWidget(self.b_input, 1, 1)

        self.classify_button = QPushButton('Classify X1,X2')
        self.classify_button.clicked.connect(self.read_input)

        self.classify_CSV_button = QPushButton('Classify from CSV file')
        self.classify_CSV_button.clicked.connect(self.read_data)

        self.save_button = QPushButton('Save to CSV file')
        self.save_button.clicked.connect(self.save)

        self.train_button = QPushButton('Train Network')
        self.train_button.clicked.connect(self.train_net)

        layout.addWidget(self.classify_button, 2, 0, 1, 2)
        layout.addWidget(self.classify_CSV_button, 3, 0, 1, 2)
        layout.addWidget(self.save_button, 4, 0, 1, 2)
        layout.addWidget(self.train_button, 5, 0, 1, 2)

        spacer1 = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer1, 6, 0, 1, 2)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel('X1')
        self.axes.set_ylabel('X2')
        self.axes.set_title('Binary Classifier')
        self.axes.set_xlim([-4,2])
        self.axes.set_ylim([2,5])
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 0, 2, 20, 20)

        self.content.setLayout(layout)
        self.setCentralWidget(self.content)

        self.canvas.mpl_connect('button_press_event', self.onclick)
        
    def onclick(self, event):
        #Write coordinates to QLineEdits if user click on canvas
        click = event.xdata, event.ydata
        if None not in click:  
            self.a_input.setText(str(event.xdata))
            self.b_input.setText(str(event.ydata))


    def read_input(self):
        #Read input from QLineEdits (a_input, b_input)
        try:
            self.x1_float = float(self.a_input.text())
            self.x2_float = float(self.b_input.text())
        except ValueError as e:
            QMessageBox.critical(self, 'Error', 'Wrong input:\nValueError: '+str(e))
            return

        if not -4 <= self.x1_float <= 2:
            QMessageBox.critical(self, 'Error', 'X1 is not in range <-4; 2>')
            return
        elif not 2 <= self.x2_float <= 5:
            QMessageBox.critical(self, 'Error', 'X2 is not in range <2; 5>')
            return
        else:
            self.input = np.array([[self.x1_float,self.x2_float]])
            window.predict()

            
    def read_data(self):
        #Read data from CSV file data.csv
        try:
            column_names = ["X1", "X2", "class"]
            #Input file for classify points
            df= pandas.read_csv('data.csv', sep = ';', decimal=",", usecols=[0,1], names=column_names, skiprows=[0])
            df = df.stack().str.replace(',','.').unstack()
        except IOError as e:
            QMessageBox.critical(self, 'Error', 'File not found:\nValueError: '+str(e))
            return
        try:
            self.x1_float = [float(i) for i in(df.X1.to_list())]
            self.x2_float = [float(i) for i in(df.X2.to_list())]
        except ValueError as e:
            QMessageBox.critical(self, 'Error', 'Wrong input from CSV file:\nValueError: '+str(e))
            return
        for i in self.x1_float:
            if not -4 <= i <= 2:
                QMessageBox.critical(self, 'Error', 'X1 is not in range <-4; 2>')
                return
        for i in self.x2_float:
            if not 2 <= i <= 5:
                QMessageBox.critical(self, 'Error', 'X2 is not in range <2; 5>')
                return

        self.input = np.array(list(zip(self.x1_float,self.x2_float)))

        window.predict()

    def show_graph(self):
        #show graph and visualize data
        x1_range = np.linspace(-4,2,100)
        x2_range = np.linspace(2,5,100)
        x1, x2 = np.meshgrid(x1_range,x2_range)
        y_plot = 0.4444444*(x1+2)**2 + 2.3668639*(x2-3)**2 -1 
        self.axes.clear()
        self.axes.contour(x1,x2,y_plot,[0])
        self.axes.grid()
        self.canvas.draw()

        group_x1_0=[]
        group_x2_0=[]
        group_x1_1=[]
        group_x2_1=[]
        if len(self.y_reshape) == 1:
            if self.y_reshape == 0:
                group_x1_0.append([self.x1_float])
                group_x2_0.append([self.x2_float])
            else:
                group_x1_1.append([self.x1_float])
                group_x2_1.append([self.x2_float])
        else: 
            for i,j,k in zip(self.y_reshape,self.x1_float,self.x2_float):
                if i == 0:
                    group_x1_0.append([j])
                    group_x2_0.append([k])
                else:
                    group_x1_1.append([j])
                    group_x2_1.append([k])
        
        self.axes.scatter(group_x1_1,group_x2_1, marker="o", color = 'blue', label = "Inside", s=30)
        self.axes.scatter(group_x1_0,group_x2_0, marker="x", color = 'red', label = "Outside", s=30)
        self.axes.set_xlabel('X1')
        self.axes.set_ylabel('X2')
        self.axes.set_title('Binary Classifier')
        self.axes.legend()
        self.axes.set_xlim([-4,2])
        self.axes.set_ylim([2,5])
        self.canvas.draw()

    def train_net(self):
        #train neural network
        reply = QMessageBox.question(self, 'Train Model', 'Are you sure you want to train model?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            #settings number of testing and evaluating points for neural network
            x1_range = np.random.uniform(-4,2,10000)
            x2_range = np.random.uniform(2,5,10000)

            training_data = [list(a) for a in zip(x1_range, x2_range)]
            y_train = 0.4444444*(x1_range+2)**2 + 2.3668639*(x2_range-3)**2 

            target_data = []
            for i in y_train:
                if i < 1:
                    target_data.append([1])
                else:
                    target_data.append([0])

            x_train, x_test, y_train, y_test = train_test_split(training_data, target_data, test_size=0.3, random_state=0)

            net = Sequential()
            net.add(Dense(16, input_dim=2, activation='tanh'))
            net.add(Dense(8, activation='tanh'))
            net.add(Dense(1, activation='sigmoid'))
            net.compile(loss='mean_squared_error', optimizer='Adam', metrics=['binary_accuracy'])
            net.fit(x_train, y_train, epochs=100, batch_size=64, verbose=0)
            test_loss, test_acc = net.evaluate(x_test, y_test)
            print('Test accuracy:', test_acc)
            print('Test loss:', test_loss)
            net.save('model.h5')
            print("Model saved to disk")
            QMessageBox.information(self, 'Train Model','Model was trained and saved.')
        else:
            return

    def predict(self):
        try:            
            net = load_model('model.h5')
        except IOError as e:
            QMessageBox.critical(self, 'Error', 'File not found:\nValueError: '+str(e))
            return
        print("\n=========== NET SUMMARY =============")
        net.summary()
        self.y = net.predict_classes(self.input)
        self.y_reshape = self.y.reshape(-1)
        print("\n=========== CLASSIFIED DATA =============")
        for i in range(len(self.input)):
            if self.y[i] == 0:
                print("X=%s, Outside" % (self.input[i]))
            else:
                print("X=%s, Inside" % (self.input[i]))
        window.show_graph()

    def save(self):
        reply = QMessageBox.question(self, 'Save Data', 'Are you sure you want to rewrite data in CSV file?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            y_class = []
            for i in self.y_reshape:
                if i == 0:
                    y_class.append('outside')
                else:
                    y_class.append('inside') 
            df = pandas.DataFrame(data={"X1":self.x1_float, "X2":self.x2_float,"class":y_class})
            df.to_csv('data.csv', sep=';', index=False)
            QMessageBox.information(self, 'Save Data','Data was succesfully saved.')
        else:
            return
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
