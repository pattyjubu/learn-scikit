import numpy as np
import matplotlib .pyplot as plt
from sklearn.linear_model import LinearRegression

# dummy train set
rng = np.random
x = rng.rand(50)*10
y = 2*x+rng.rand(50)

# linear regression model
model = LinearRegression()

# train model
# x argument must be 2D array[[1],[2], need to reshape befo
x_new = x.reshape(-1, 1)
model.fit(x_new, y)

# evaluate model
print("Intercept: {}".format(model.intercept_))
print("Coefficient: {}".format(model.coef_))
print("R-SQuare: {}".format(model.score(x_new, y)))

# create test set
xfit = np.linspace(-1, 11)

# test model
xfit_new = xfit.reshape(-1, 1)
yfit = model.predict(xfit_new)

# analysis model & result
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
