import pylab
import matplotlib.pyplot as plt
from sklearn import datasets

digits_dataset = datasets.load_digits()

print(digits_dataset.target[0])
# pylab.imshow(digits_dataset.images[0], cmap=pylab.cm.gray_r)
# pylab.show()

plt.imshow(digits_dataset.images[0], cmap=plt.get_cmap('gray'))
plt.show()
