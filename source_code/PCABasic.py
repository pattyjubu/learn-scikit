from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Generate grouping data with 10 features (columns)
x, y = make_blobs(n_samples=100,n_features=10)
print('Before transform = ', x.shape)

# model PCA (not for predicted, just reduce feature to 4 features)
pca = PCA(n_components=4)
x_pca = pca.fit_transform(x)
print('After transform = ', x_pca.shape)

df = pd.DataFrame({'var': pca.explained_variance_ratio_,'pc': ['PC1', 'PC2', 'PC3', 'PC4']})
sb.barplot(x='pc', y='var', data=df, color='c')
print(pca.explained_variance_ratio_)
plt.show()