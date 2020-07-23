import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data_set', 'mnist-original.mat')
mnist_raw = loadmat(mnist_file_path)

mnist = {
    'data': mnist_raw['data'].T, #.T is transpose
    'target': mnist_raw['label'][0]
}

x, y = mnist['data'], mnist['target']

shown_index = 15000
print(y[shown_index])
plt.imshow(
    x[shown_index].reshape(28,28),
    cmap=plt.cm.binary,
    interpolation="nearest"
)
plt.show()