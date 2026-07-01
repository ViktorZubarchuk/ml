import pandas as pd

dataset = pd.read_csv('cars_dataset.csv', sep=';')
df = pd.DataFrame(data = dataset, columns = dataset.columns)

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
    print(f"Название: price_rub")
    print(f"Минимум: {df['price_rub'].min():.3f}")
    print(f"Максимум: {df['price_rub'].max():.3f}")
    print(f"Среднее: {df['price_rub'].mean():.3f}")
    print(f"Медиана: {df['price_rub'].median():.3f}")
    print(f"Стандартное отклонение: {df['price_rub'].std():.3f}")


if __name__ == "__main__":
    main_info(df)