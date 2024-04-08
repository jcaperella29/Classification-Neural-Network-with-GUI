# Reading the cleaned numeric titanic survival data
import pandas as pd
import numpy as np
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense

from sklearn import metrics
import shutil
import PySimpleGUI as sg
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile
import os 
import copy




def run_NN(pickle_path) :
# To remove the scientific notation from numpy arrays
    np.set_printoptions(suppress=True)

    TitanicSurvivalDataNumeric=pd.read_pickle(pickle_path)
    TitanicSurvivalDataNumeric.head()




    TargetVariable=['Survived']
    Predictors=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_C', 'Embarked_Q', 'Embarked_S']
 
    X=TitanicSurvivalDataNumeric[Predictors].values
    y=TitanicSurvivalDataNumeric[TargetVariable].values
 
 
### Sandardization of data ###
### We do  not standardize the Target variable for classification
    from sklearn.preprocessing import StandardScaler    
    PredictorScaler=StandardScaler()
 
# Storing the fit object for later reference
    PredictorScalerFit=PredictorScaler.fit(X)
 
# Generating the standardized values of X and y
    X=PredictorScalerFit.transform(X)
 
# Split the data into training and testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Quick sanity check with the shapes of Training and Testing datasets
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    classifier = Sequential()
# Defining the Input layer and FIRST hidden layer,both are same!
# relu means Rectifier linear unit function
    classifier.add(Dense(units=10, input_dim=9, kernel_initializer='uniform', activation='relu'))

#Defining the SECOND hidden layer, here we have not defined input because it is
# second layer and it will get input as the output of first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Defining the Output layer
# sigmoid means sigmoid activation function
# for Multiclass classification the activation ='softmax'
# And output_dim will be equal to the number of factor levels
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Optimizer== the algorithm of SGG to keep updating weights
# loss== the loss function to measure the accuracy
# metrics== the way we will compare the accuracy after each step of SGD
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the Neural Network on the training data
    survivalANN_Model=classifier.fit(X_train,y_train, batch_size=10 , epochs=10, verbose=1)

# Defining a function for finding best hyperparameters
    def FunctionFindBestParams(X_train, y_train):
    
    # Defining the list of hyper parameters to try
        TrialNumber=0
        batch_size_list=[5, 10, 15, 20]
        epoch_list=[5, 10, 50 ,100]
    
        import pandas as pd
        SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
        for batch_size_trial in batch_size_list:
            for epochs_trial in epoch_list:
                TrialNumber+=1
            
            # Creating the classifier ANN model
                classifier = Sequential()
                classifier.add(Dense(units=10, input_dim=9, kernel_initializer='uniform', activation='relu'))
                classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
                classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
                classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            
                survivalANN_Model=classifier.fit(X_train,y_train, batch_size=batch_size_trial , epochs=epochs_trial, verbose=0)
            # Fetching the accuracy of the training
                Accuracy = survivalANN_Model.history['accuracy'][-1]
            
            # printing the results of the current iteration
                temp=pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), Accuracy]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] )
                SearchResultsData=pd.concat([SearchResultsData,temp])
                return(SearchResultsData)
 
    

###############################################

# Calling the function
    ResultsData=FunctionFindBestParams(X_train, y_train)

# Training the model with best hyperparamters
    classifier.fit(X_train,y_train, batch_size=5 , epochs=100, verbose=1)


# Predictions on testing data
    Predictions=classifier.predict(X_test)

# Scaling the test data back to original scale
    Test_Data=PredictorScalerFit.inverse_transform(X_test)

# Generating a data frame for analyzing the test data
    TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
    TestingData['Survival']=y_test
    TestingData['PredictedSurvivalProb']=Predictions

# Defining the probability threshold
    def probThreshold(inpProb):
                if inpProb  >  0.5:
                    return(1)
                else:
                    return(0)

# Generating predictions on the testing data by applying probability threshold
    TestingData['PredictedSurvival']=TestingData['PredictedSurvivalProb'].apply(probThreshold)



###############################################






    report= { 'predicted'  :TestingData['PredictedSurvival']}
    report_df=pd.DataFrame(report)


    
    


    fpr,tpr,thresholds=metrics.roc_curve(TestingData['Survival'], TestingData['PredictedSurvival'],pos_label= 1)
    AUC=metrics.auc(fpr, tpr)
    report_df['AUC']=pd.Series()
    report_df.iloc[0,report_df.columns.get_loc('AUC')]=AUC


   

    

    

    return(report_df)

#front end
sg.theme('Reddit')
layout =  [ [sg.Text("Classify data using a neural network"), sg.InputText(key="_file_"), sg.FileBrowse(target="_file_"),[sg.Submit()],[sg.Cancel()]]]
        
newlayout = copy.deepcopy(layout)
window = sg.Window('Select a pickle file with your data ', newlayout, size=(270*4,4*100))
event, values = window.read()
pickle_path_path = None
while True:
    event, values = window.read()
    print(event, values)
    
    if event == 'Cancel':
        break
    elif event == 'Submit':
        
        pickle_path= values['_file_']
        if pickle_path:
            results=run_NN(pickle_path)
        break
window.close()


def save_file():
            file = filedialog.asksaveasfilename(
                
            filetypes=[("csv file", ".csv")],
            defaultextension=".csv",
            title='Save Output')
            results_file=results.to_csv(str(file))
            if file: 
                            fob=open(str(results_file),'w')
                            fob.write("Save results")
                            fob.close()
            else: # user cancel the file browser window
                        print("No file chosen")

        
if results is not None:        
        my_w = tk.Tk()
        my_w.geometry("400x300")  # Size of the window 
        my_w.title('Save results ')
        my_font1=('times', 18, 'bold')
        l1 = tk.Label(my_w,text='Save',width=30,font=my_font1)
        l1.grid(row=1,column=1)
        
        b1 = tk.Button(my_w, text='Save', 
        width=20,command = lambda:save_file())
        b1.grid(row=2,column=1)
        
        
        
       
        b3=tk.Button(my_w, text="Quit", command=my_w.destroy)
        b3.grid(row = 3, column=1)    
        my_w.mainloop()  # Keep the window open



# Orginal code came from https://thinkingneuron.com/how-to-use-artificial-neural-networks-for-classification-in-python/ , I made changes to add in reporting and adding a GUI