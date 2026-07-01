from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

def main_info(dataset):
    print("ОСНОВНАЯ ИНФОРМАЦИЯ О ДАННЫХ")
    
    print("Первые 5 строк:")
    print(dataset.head())
    
    print("\nОбщая информация:")
    dataset.info()
    
    print("\nСтатистическое описание:")
    print(dataset.describe())
    
    print("\nКоличество пропущенных значений:")
    print(dataset.isnull().sum())

if __name__ == "__main__":
    main_info(iris_df)
