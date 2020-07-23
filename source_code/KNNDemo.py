from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.4, random_state=0)

# model
knn = KNeighborsClassifier(n_neighbors=1)

# training
knn.fit(x_train, y_train)

# predict
y_pred = knn.predict(x_test)

# print('predict result', pred)
# print('Class', iris_dataset['target_names'][pred])

print(classification_report(y_test, y_pred, target_names=iris_dataset['target_names']))
print('accuracy', accuracy_score(y_test, y_pred)*100)
