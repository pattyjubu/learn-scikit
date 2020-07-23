import seaborn as sb
import matplotlib.pyplot as plt

iris_dataset = sb.load_dataset("iris")
#  show first 5 lines
print(iris_dataset.head())

sb.set()
sb.pairplot(iris_dataset, hue="species", height=2)
plt.show()