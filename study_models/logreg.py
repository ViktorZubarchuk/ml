import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss
 
np.random.seed(42)
n_samples = 100
n_features = 3
true_w = [3.0, 2.0, 1.0]
true_b = 

