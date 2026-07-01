import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

dataset = pd.read_csv('cars_dataset.csv', sep = ';')
df = pd.DataFrame(data = dataset, columns = dataset.columns)

X = df.drop('price_rub', axis=1)
y = df['price_rub']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cat_fea = ['brand', 'model', 'transmission', 'body_type', 'color']
num_fea = df.columns.drop(cat_fea + ['price_rub'])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat_prep', cat_pipe, cat_fea),
    ('num_prep', num_pipe, num_fea)
])


models = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {
            'model__fit_intercept': [True, False],
        }
    },
    'Lasso': {
        'model': Lasso(),
        'params': {
            'model__alpha':[0.001, 0.01, 0.1, 1],
            'model__max_iter': [100, 500, 1000],
            'model__fit_intercept':[True, False],
            'model__tol': [0.0001, 0.0005, 0.001]
        }
    },
    'Ridge': {
        'model': Ridge(),
        'params': {
            'model__alpha':[0.001, 0.01, 0.1, 1],
            'model__max_iter': [1000, 2000, 3000],
            'model__fit_intercept':[True, False],
            'model__tol': [0.0001, 0.0005, 0.001],
        }
    },
    'SVR': {
        'model': SVR(),
        'params':{
            "model__kernel": ["linear", "rbf", "poly"],
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"]  
        }
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params':{
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]  # Manhattan / Euclidean
        }
    },
    'DecisionTreeRegressor':{
        'model': DecisionTreeRegressor(),
        'params': {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        }
    },
    'RandomForestRegressor':{
        'model': RandomForestRegressor(),
        'params': {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"]
        }
    },
    'GradientBoostingRegressor':{
        'model': GradientBoostingRegressor(),
        'params': {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        }
    }
}

results = {}
best_model_name = None
best_r2 = -np.inf

for name, config in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', config['model'])
    ])
    grid = GridSearchCV(estimator = pipe, param_grid = config['params'], scoring = 'r2', cv = 5, n_jobs=4)
    grid.fit(X_train, y_train)
    
    results[name] = {
        'best_score': grid.best_score_,
        'best_params': grid.best_params_
    }

    y_pred = grid.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)

    print(f"\n\nMODEL: {name}")
    print(f"GRID - Best R2: {grid.best_score_:.4f}")
    print(f"GRID - Best params: {grid.best_params_}")
    print(f"TEST - R2: {test_r2:.4f}")
    print(f"TEST - MAE: {test_mae:.4f}")
    print(f"TEST - MSE: {test_mse:.4f}")

    if test_r2 > best_r2:
        best_r2 = test_r2
        best_model_name = grid.best_estimator_
        best_grid = grid

print(f"\n\nЛучшая модель: {best_model_name}")
print(f"TEST - R2: {best_r2:.4f}")

print("\n\nСравнение всех моделей:")
for name, metrics in results.items():
    print(f"{name}: R2 = {metrics['best_score']:.4f}")

final_model = best_grid.best_estimator_
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

final_model.fit(X_full, y_full)

joblib.dump(final_model, 'best_model.pkl')