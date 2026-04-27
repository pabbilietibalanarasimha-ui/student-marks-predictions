import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

X = data[['Hours']]
y = data['Marks']

model = LinearRegression()
model.fit(X, y)

pred = model.predict([[5]])
print("Predicted Marks:", pred)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()