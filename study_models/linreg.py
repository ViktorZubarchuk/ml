import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
n_samples = 1000
n_features = 2
true_w = np.array([2.5, -1.5])
true_b = 4.0
noise = np.random.uniform(-0.5, 0.5, n_samples)

X = np.random.randn(n_samples, n_features) * 10
y = X @ true_w + true_b + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model1 = LinearRegression(fit_intercept = True, positive = False)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('\n\nLinearRegression(fit_intercept = True, positive = False)')
print(f'mae = {mae:.2f}\nmse = {mse:.2f}\nrmse = {np.sqrt(mse):.2f}\nr2 = {r2:.2f}')
print(f'[w1, w2] = {model1.coef_}\nb = {model1.intercept_}')

model2 = LinearRegression(fit_intercept = False, positive=False)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('\n\nLinearRegression(fit_intercept = False, positive = False)')
print(f'mae = {mae:.2f}\nmse = {mse:.2f}\nrmse = {np.sqrt(mse):.2f}\nr2 = {r2:.2f}')
print(f'[w1, w2] = {model2.coef_}\nb = {model2.intercept_}')

model3 = LinearRegression(fit_intercept = True, positive = True)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('\n\nLinearRegression(fit_intercept = True, positive = True)')
print(f'mae = {mae:.2f}\nmse = {mse:.2f}\nrmse = {np.sqrt(mse):.2f}\nr2 = {r2:.2f}')
print(f'[w1, w2] = {model3.coef_}\nb = {model3.intercept_}')

print(f'\nTrue w = {true_w}\nTrue b = {true_b}')


'''
fit_intercept - надо ли добавить свободный член(b) в модель

positive - все коэффициенты при признаках(w) принудительно делаются неотрицательными
'''