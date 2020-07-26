import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def clean_data(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

def split_feature_class(dataset, feature):
    features = dataset.drop(feature, axis=1) # get all data except "income"
    labels = dataset[feature].copy() # get only "income" data
    return features, labels

adult_data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data_set', 'adult.csv')

dataset = pd.read_csv(adult_data_file_path)
dataset = clean_data(dataset)

# Split train and test set
train_set, test_set = train_test_split(dataset, test_size=0.2)
train_features, train_labels = split_feature_class(train_set, 'income')
test_features, test_labels = split_feature_class(test_set, 'income')

# model
model = GaussianNB()
model.fit(train_features, train_labels)
clf_pred = model.predict(test_features)

print('Accuracy = ', accuracy_score(test_labels, clf_pred))

