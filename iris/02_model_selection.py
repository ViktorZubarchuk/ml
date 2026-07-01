import joblib
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris['target']

X = iris_df.drop('target', axis = 1)
y = iris_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params':{
            'model__max_iter':[100, 500, 1000]   
        }
    },
    'SVC': {
        'model': SVC(),
        'params':{
            "model__kernel": ["linear", "rbf", "poly"],
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"]  
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params':{
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]  # Manhattan / Euclidean
        }
    },
    'DecisionTreeClassifier':{
        'model': DecisionTreeClassifier(),
        'params': {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__criterion": ['gini', 'entropy', 'log_loss']
        }
    },
    'RandomForestClassifier':{
        'model': RandomForestClassifier(),
        'params': {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"]
        }
    },
    'GradientBoostingClassifier':{
        'model': GradientBoostingClassifier(),
        'params': {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        }
    }
}

results = {}
best_model_name = None
best_accuracy = -np.inf

for name, config in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', config['model'])
    ])
    grid = GridSearchCV(pipe, config['params'], scoring='accuracy', n_jobs = 4, cv = 5)
    grid.fit(X_train, y_train)
    results[name] = {
        "best_score": grid.best_score_,
        "best_params": grid.best_params_
    }

    y_pred = grid.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_report = classification_report(y_test, y_pred)
    
    print(f"\n\n\n\nMODEL: {name}")
    print(f"GRID - Best accuracy: {grid.best_score_:.4f}")
    print(f"GRID - Best params: {grid.best_params_}")
    print(f"TEST - accuracy: {test_accuracy:.4f}")
    print(f"TEST - report:\n {test_report}")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_name = name
        best_grid = grid

print(f"\n\nЛучшая модель: {best_model_name}")
print(f"TEST - accuracy: {best_accuracy:.4f}")

print("\n\nСравнение всех моделей:")
for name, metrics in results.items():
    print(f"{name}: accuracy = {metrics['best_score']:.4f}")


final_model = best_grid.best_estimator_
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

final_model.fit(X_full, y_full)

joblib.dump(final_model, 'iris/best_model.pkl')
