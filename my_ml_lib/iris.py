from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as sklearnKnn
from models import KNN as myKnn
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

np.random.seed(42)

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target']= data.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sk_model = sklearnKnn()
my_model = myKnn()

my_model.fit(X_train_scaled, y_train)
sk_model.fit(X_train_scaled, y_train)

my_model_y_pred = my_model.predict(X_test)
sk_model_y_pred = sk_model.predict(X_test)

print('MY_MODEL')
print(f'accuracy_score = {accuracy_score(y_test, my_model_y_pred)}')

print('SK_MODEL')
print(f'accuracy_score = {accuracy_score(y_test, sk_model_y_pred)}')
