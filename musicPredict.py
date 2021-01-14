# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# program uses a made up data set to predict what music genre a person will like using their gender and age as arguments


import pandas as pd #imports pandas
from sklearn.tree import DecisionTreeClassifier #imports sklearn from machine learning library, implements decision tree algorithm
from sklearn.model_selection import train_test_split #uses to easily split data set for traning and testing
from sklearn import tree #exports decision tree in graphical format

# from sklearn.metrics import accuracy_score #for accuracy score between 0-1
import joblib #stores trained model , has methods for saving and loading models

music_data = pd.read_csv('music.csv') #puts data set into variable music_data
X = music_data.drop(columns=['genre']) #splits data set adds a new table from music_data without the genre column (input set)
y = music_data['genre'] #splits data set adds a new table from music_data with only genre column (output set)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier() #creates model to train
model.fit(X, y) #takes parameters for training data

tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names=['age', 'gender'], #to see rules in nodes
                    class_names=sorted(y.unique()), #displays class for each node
                    label = 'all', #every node has labels
                    rounded=True, #rounded corners for graph
                    filled=True) #boxes/nodes are filled with a color

# model = joblib.load('music-recommender.joblib') #uses joblib to make training easier

# predictions = model.predict([[21,1]]) #uses arguments to create prediction


# score = accuracy_score(y_test, predictions)



