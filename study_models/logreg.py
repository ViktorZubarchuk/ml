import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss
 
np.random.seed(42)
n_samples = 100
repetitor_prep = np.random.randint(0, 6, n_samples)  # часы с репетитором
self_perp = np.random.randint(0, 6, n_samples)     # часы самостоятельных занятий
age = np.random.randint(18, 23, n_samples)  # возраст студента
true_w = np.array([3.0, 2.0, 1.0, 0.0, 0.0])  # репетитор важнее чем сам, возраст еще менее важен
true_b = 1.0
noise = np.random.normal(0, 2, n_samples) 

X = pd.DataFrame({
    'репетитор': repetitor_prep,
    'сам': self_perp,
    'возраст': age
})
# Шумовые признаки (не влияют на y)
X['пол'] = np.random.randint(0, 2, n_samples)
X['спорт'] = np.random.randint(0, 2, n_samples)

y_score = X @ true_w + true_b + noise
y = (y_score >= (max(y_score)+min(y_score))/2.2 ).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid_logreg = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    
}
logreg = LogisticRegression()
