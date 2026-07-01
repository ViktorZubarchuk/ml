import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def resume_model(name, best_model, y_test, y_pred, grid = None):
    print(f'\n\n{name}')
    # print_coefs(best_model)
    print_metrics(y_test, y_pred)
    if name != "LinearRegression":
        print(f'Best parameters: {grid.best_params_}')
        print(f'Best score (r2) = {grid.best_score_:.4f}')

def print_coefs(best_model):
    print('Coef:')
    print('-' * 30)
    print(f'{"Feature":<8} {"Value":>10} {"True":>10}')
    print('-' * 30)
    for i, (w_pred, w_true) in enumerate(zip(best_model.coef_, true_w)):
        print(f'w{i+1:<7} {w_pred:>10.3f} {w_true:>10.1f}')
    print(f'b{" ":<7} {best_model.intercept_:>10.3f} {true_b:>10.2f}')
    print('-' * 30)

def print_original_coefs(model, scaler, true_w, true_b, name):
    """Выводит коэффициенты модели в исходном масштабе"""
    w_original = model.coef_ / scaler.scale_
    b_original = model.intercept_ - np.sum(w_original * scaler.mean_)
    
    print(f'\n{name} (исходный масштаб):')
    print(f'{"Feature":<8} {"Value":>10} {"True":>10}')
    print('-' * 30)
    for i, (w_pred, w_true) in enumerate(zip(w_original, true_w)):
        print(f'w{i+1:<7} {w_pred:>10.3f} {w_true:>10.1f}')
    print(f'b{" ":<7} {b_original:>10.3f} {true_b:>10.2f}')
    print('-' * 30)

def print_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'mae = {mae:.2f}\nmse = {mse:.2f}\nrmse = {rmse:.2f}\nr2 = {r2:.2f}')

np.random.seed(42)
n_samples = 100
n_features = 50
n_informative = 5
true_w = np.zeros(n_features)
true_w[:n_informative] = np.array([3.0, 2.5, -2.0, 1.5, -1.0])
true_b = 4.0
noise = np.random.randn(n_samples) * 5.0

# Коррелированные признаки
cov = np.eye(n_features)
for i in range(n_features):
    for j in range(n_features):
        if i != j:
            cov[i, j] = 0.85

X = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples) * 10
y = X @ true_w + true_b + noise

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)



# Lasso
param_grid_lasso = {
    'alpha': [0.01, 0.1, 1, 10],
    'max_iter': [10000, 15000],
    'tol': [1e-5, 1e-4, 1e-3],
    'selection': ['cyclic', 'random'],
    'warm_start': [True, False]
}

lasso = Lasso()
grid = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='r2')
grid.fit(X_train, y_train)
best_lasso = grid.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)

# Ridge
param_grid_ridge = {
    'alpha': [0.01, 0.1, 1, 10],
    'max_iter': [10000, 15000],
    'tol': [1e-5, 1e-4, 1e-3]
}

ridge = Ridge()
grid = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='r2')
grid.fit(X_train, y_train)
best_ridge = grid.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)

# Elasticnet
param_grid_elasticnet = {
    'alpha': [0.01, 0.1, 1, 10],
    'max_iter': [10000, 15000],
    'tol': [1e-5, 1e-4, 1e-3],
    'selection': ['cyclic', 'random'],
    'warm_start': [True, False],
    'l1_ratio': [0.5, 0.6, 0.7, 0.8, 0.9]
}

elasticnet = ElasticNet()
grid = GridSearchCV(elasticnet, param_grid_elasticnet, cv=5, scoring='r2')
grid.fit(X_train, y_train)
best_elasticnet = grid.best_estimator_
y_pred_elasticnet = best_elasticnet.predict(X_test)

# Linreg
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)



print('\n' + '='*50)
print('СРАВНЕНИЕ МОДЕЛЕЙ (все на стандартизированных данных)')
print('='*50)

# Lasso
resume_model("Lasso", best_lasso, y_test, y_pred_lasso, grid)

# Ridge
resume_model("Ridge", best_ridge, y_test, y_pred_ridge, grid)

# Elasticnet
resume_model("Elasticnet", best_elasticnet, y_test, y_pred_elasticnet, grid)

# LinearRegression
resume_model("LinearRegression", best_model=linreg, y_test = y_test, y_pred = y_pred_elasticnet)




print('\n' + '='*50)
print('ВОССТАНОВЛЕННЫЕ КОЭФФИЦИЕНТЫ (исходный масштаб)')
print('='*50)

# Восстанавливаем Lasso
print_original_coefs(best_lasso, scaler, true_w, true_b, 'Lasso')

# Восстанавливаем Ridge
print_original_coefs(best_ridge, scaler, true_w, true_b, 'Ridge')

# Восстанавливаем Elasticnet
print_original_coefs(best_elasticnet, scaler, true_w, true_b, 'ElasticNet')

# Восстанавливаем LinearRegression
print_original_coefs(linreg, scaler, true_w, true_b, 'LinearRegression')





'''
Lasso - линейная регрессия с L1-регуляризацией - штрафом на сумму абсолютных значений коэффициентов.
Сжимает коэффициенты к нулю, зануляет неважные признаки (отбор признаков).

Ridge - линейная регрессия с L2-регуляризацией - штрафом на сумму квадратов коэффициентов.
Сжимает коэффициенты к нулю, но не зануляет их полностью. 
Равномерно распределяет вес между коррелированными признаками.

ElasticNet - линейная регрессия с комбинированной L1+L2 регуляризацией.
Объединяет свойства Lasso и Ridge: зануляет признаки как Lasso и стабилизирует как Ridge.
Параметр l1_ratio управляет балансом: 1 - Lasso, 0 - Ridge.



fit_intercept - надо ли добавить свободный член (b) в модель

positive - все коэффициенты при признаках (w) принудительно делаются неотрицательными

alpha - сила регуляризации. Чем больше, тем сильнее штраф и проще модель

max_iter - максимальное число итераций. Увеличивайте, если алгоритм не сходится

tol - точность останова. Меньше → точнее, но дольше

selection - порядок обновления весов: 'cyclic' — по порядку, 'random' — случайно

warm_start - если True, использует веса с прошлого fit как начальные для ускорения (удобно при переборе близких alpha)



Вывод:
Для моих данных (много шумовых признаков + корреляция 0.85):

Lasso — лучший выбор, т.к. отбор признаков (зануляет), высокая точность (для моих данных самое лучшее, так как нулевых очень много)

ElasticNet — хороший компромисс если нужно больше стабильности (для моих данных не очень так как нулевых очень много)

Ridge — если нужна стабильность и все признаки важны (для моих данных не очень так как нулевых очень много)

LinearRegression — лучше не использовать, она использует все признаки (не занулила) и переобучилась

'''