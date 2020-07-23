from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

diabetes_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data_set', 'diabetes.csv')
df = pd.read_csv(diabetes_file_path)
x = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# # find k to model
# k_neighbors = np.arange(1, 9) #[1,2,3,...,9]
# train_score = np.empty(len(k_neighbors))
# test_score = np.empty(len(k_neighbors))
#
# for i, k in enumerate(k_neighbors):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train, y_train)
#     train_score[i] = knn.score(x_train, y_train)
#     test_score[i] = knn.score(x_test, y_test)
#
# plt.title('Compare of k value in model')
# plt.plot(k_neighbors, test_score, label='Test Score')
# plt.plot(k_neighbors, train_score, label='Train Score')
# plt.legend()
# plt.xlabel('K number')
# plt.ylabel('Score')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred))

#confusion_matrix => TruePositive,FalsePositive,FalseNeg,TrueNeg
print(confusion_matrix(y_test, y_pred))
# or can use crosstab
print(pd.crosstab(y_test, y_pred, rownames=['Actually'], colnames=['Prediction'], margins=True))