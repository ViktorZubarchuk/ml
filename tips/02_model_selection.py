import joblib
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

tips = sns.load_dataset("tips")

y = tips["tip"]
X = tips.drop(columns = ["tip"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

num_pipe = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

models = {
    "LinearRegression": LinearRegression(),
    "SVM": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor()
}
lr_params = {
    "model__fit_intercept": [True, False]
}
svr_params = {
    "model__kernel": ["linear", "rbf", "poly"],
    "model__C": [0.1, 1, 10, 100],
    "model__epsilon": [0.01, 0.1, 0.2],
    "model__gamma": ["scale", "auto"]
}
knn_params = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2]  # Manhattan / Euclidean
}
decision_tree_params = {
    "model__max_depth": [3, 5, 7, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__criterion": ["squared_error", "friedman_mse", "absolute_error"]
}
random_forest_params = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}
gradient_boosting_params = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 1.0],
    "model__loss": ["squared_error", "huber"]
}
param_grids = {
    "LinearRegression": lr_params,
    "SVM": svr_params,
    "KNeighborsRegressor": knn_params,
    "DecisionTreeRegressor": decision_tree_params,
    "RandomForestRegressor": random_forest_params,
    "GradientBoostingRegressor": gradient_boosting_params
}

results = {}
best_model_name = None
best_r2 = -np.inf
for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    grid = GridSearchCV(estimator=pipe, param_grid=param_grids[name], cv=5,scoring="r2",n_jobs=-1)
    grid.fit(X_train, y_train)

    results[name] = {
        "best_score": grid.best_score_,
        "best_params": grid.best_params_
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
        best_model_name = name
        best_grid = grid

def plot_results(results):    
    names = list(results.keys())
    r2_scores = [results[name]['best_score'] for name in names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, r2_scores)
    plt.axhline(y=best_r2, color='r', linestyle='--', label='Лучший результат')
    plt.xlabel('Модель')
    plt.ylabel('R2 (кросс-валидация)')
    plt.title('Сравнение качества моделей')
    plt.xticks(rotation=45)
    
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

print(f"\n\nЛучшая модель: {best_model_name}")
print(f"TEST - R2: {best_r2:.4f}")

print("\n\nСравнение всех моделей:")
for name, metrics in results.items():
    print(f"{name}: R2 = {metrics['best_score']:.4f}")

plot_results(results)

final_model = best_grid.best_estimator_
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

final_model.fit(X_full, y_full)

joblib.dump(final_model, 'tips/best_model.pkl')

