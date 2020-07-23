import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score # got score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict # got predict result
import numpy as np
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def display_image(x):
    plt.imshow(
        x.reshape(28, 28),
        cmap=plt.cm.binary,
        interpolation='nearest'
    )
    plt.show()


def display_predict(clf, actual_y, x):
    print('Actual = ', actual_y)
    print('Predict is 0 = ', clf.predict([x])[0])


def displayConfusionMatrix(cm,cmap=plt.cm.GnBu):
    classes=["Other Number","Number 5"]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes))
    plt.xticks(trick_marks,classes)
    plt.yticks(trick_marks,classes)
    thresh=cm.max()/2
    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),
        horizontalalignment='center',
        color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

mnist_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data_set', 'mnist-original.mat')
mnist_raw = loadmat(mnist_file_path)

mnist = {
    'data': mnist_raw['data'].T, #.T is transpose
    'target': mnist_raw['label'][0]
}

# Split train and test set
x, y = mnist['data'], mnist['target']
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# classify 0 or not
predict_idx = 500
y_train_0 = (y_train == 0) # y_train_0 = [true, false, ..., false]
y_test_0 = (y_test == 0)

# Train data
sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train_0)

# display_predict(sgd_clf, y_test[predict_idx], x_test[predict_idx])
# display_image(x_test[predict_idx])

# score = cross_val_score(sgd_clf, x_train, y_train_0, cv=3, scoring='accuracy') # cv=3, iterate 3 times
# print(score)

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_0, cv=3)
cm = confusion_matrix(y_train_0, y_train_pred) # compare real y and predicted y
print(cm)

# plt.figure()
# displayConfusionMatrix(cm)

y_test_pred = sgd_clf.predict(x_test)
classes = ['OtherNumber', 'Number 0']
print(classification_report(y_test_0, y_test_pred, target_names=classes))
print('Accuracy Score = ', accuracy_score(y_test_0, y_test_pred)*100)