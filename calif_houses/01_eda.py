import pandas as pd
from sklearn.datasets import fetch_california_housing

'''
Описание признаков датасета California Housing:
MedInc    - медианный доход жителей района (в десятках тысяч долларов США).
HouseAge  - медианный возраст домов в районе (в годах).
AveRooms  - среднее количество комнат на одно домохозяйство.
AveBedrms - среднее количество спален на одно домохозяйство.
Population - численность населения района.
AveOccup  - среднее количество жителей на одно домохозяйство.
Latitude  - географическая широта района.
Longitude - географическая долгота района.

MedHouseVal(target) - медианная стоимость жилья в районе
            (в сотнях тысяч долларов США).
'''

dataset = fetch_california_housing()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["target"] = dataset.target


def main_info(df):
    print("ОСНОВНАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")

    print("\n1. Первые 5 строк:")
    print(df.head())

    print("\n2. Размер датасета:")
    print(f"Строк: {df.shape[0]}")
    print(f"Столбцов: {df.shape[1]}")

    print("\n3. Названия столбцов:")
    print(df.columns.tolist())

    print("\n4. Общая информация:")
    df.info()

    print("\n5. Статистическое описание:")
    print(df.describe())

    print("\n6. Количество пропущенных значений:")
    print(df.isnull().sum())

    print("\n7. Информация о целевой переменной:")
    print(f"Название: target")
    print(f"Минимум: {df['target'].min():.3f}")
    print(f"Максимум: {df['target'].max():.3f}")
    print(f"Среднее: {df['target'].mean():.3f}")
    print(f"Медиана: {df['target'].median():.3f}")
    print(f"Стандартное отклонение: {df['target'].std():.3f}")


if __name__ == "__main__":
    main_info(df)