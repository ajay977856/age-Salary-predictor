import pickle
from sklearn.linear_model import LinearRegression

X = [[25, 30000], [35, 50000], [45, 80000]]
y = [1, 2, 3]

model = LinearRegression()
model.fit(X, y)

with open('mymodel.pkl', 'wb') as file:
    pickle.dump(model, file)
    