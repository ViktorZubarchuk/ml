import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n_samples = 300
x1 = np.random.uniform(-3,3,n_samples)
x2 = np.sin(2 * x1) + np.random.randn(n_samples) * 0.5
X = np.column_stack([x1,x2])
y = (x1**2 + np.random.randn(n_samples)*0.3 > np.sin(2 * x1) + 0.5).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

param_grid_svc = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10],
    'gamma': ['scale', 'auto'],
    'tol': [1e-3,1e-4,1e-5],
    'max_iter': [-1, 5000, 10000, 15000],
    'decision_function_shape': ['ovo', 'ovr']
}

svc = SVC(kernel='rbf', probability=True)
grid = GridSearchCV(svc, param_grid_svc, cv=5, scoring = 'accuracy')
grid.fit(X_train_scaled, y_train)

best_svc = grid.best_estimator_
y_pred = best_svc.predict(X_test)

class_rep = classification_report(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

print('\n\nSVC')
print(f'classification_report:\n{class_rep}')
print(f'accuracy_score = {acc_score:.2f}')
print(f'grid best accuracy_score = {grid.best_score_:.2f}')
print(f'grid best params = {grid.best_params_}')


