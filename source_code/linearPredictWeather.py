import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data_set', 'Weather.csv')
dataset = pd.read_csv(dataset_file_path)

print(dataset.describe())

# dataset.plot(x='MinTemp', y='MaxTemp', style='o')
# plt.title('Min&Max temp')
# plt.xlabel('Mintemp')
# plt.ylabel('Maxtemp')
# plt.show()

# train & test set
x = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)

# split train and test set 80%:20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# training
model = LinearRegression()
model.fit(x_train, y_train)

# testing
y_pred = model.predict(x_test)

# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_pred, color='red', linewidth=2)
# plt.show()

# compare true data & predict data
df = pd.DataFrame({'Actually': y_test.flatten(), 'Predicted': y_pred.flatten()})

# df1 = df.head(20)
# df1.plot(kind='bar', figsize=(16, 10))
# plt.show()

print('MAE = ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE = ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE = ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Score = ', metrics.r2_score(y_test, y_pred))
