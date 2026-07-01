import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

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


def basic_statistics(dataset):
    print("\n\nОСНОВНЫЕ СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ")
    
    print(f"Чаевые:")
    print(f"  Средние:   {dataset['tip'].mean():.2f}")
    print(f"  Максимум:  {dataset['tip'].max():.2f}")
    print(f"  Минимум:   {dataset['tip'].min():.2f}")
    print(f"  Медиана:   {dataset['tip'].median():.2f}")
    
    print(f"\nСчета:")
    print(f"  Средний:   {dataset['total_bill'].mean():.2f}")
    print(f"  Максимум:  {dataset['total_bill'].max():.2f}")
    print(f"  Минимум:   {dataset['total_bill'].min():.2f}")
    print(f"  Медиана:   {dataset['total_bill'].median():.2f}")
    
    # Процент чаевых
    dataset['tip_percent'] = (dataset['tip'] / dataset['total_bill']) * 100
    print(f"\nПроцент чаевых от счета:")
    print(f"  Средний:   {dataset['tip_percent'].mean():.2f}%")
    print(f"  Максимум:  {dataset['tip_percent'].max():.2f}%")
    print(f"  Минимум:   {dataset['tip_percent'].min():.2f}%")



def group_statistics(dataset):
    print("\n\nСТАТИСТИКА ПО ГРУППАМ")
    
    print("По полу:")
    print(dataset.groupby("sex", observed=True)["tip"].mean().round(2))
    
    print("\nПо курению:")
    print(dataset.groupby("smoker", observed=True)["tip"].mean().round(2))
    
    print("\nПо дню недели:")
    print(dataset.groupby("day", observed=True)["tip"].mean().round(2))
    
    print("\nПо времени:")
    print(dataset.groupby("time", observed=True)["tip"].mean().round(2))


def visualize_data(dataset):
    print("\n\nВИЗУАЛИЗАЦИЯ ДАННЫХ")
    
    # График 1: Распределение чаевых
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    dataset['tip'].hist(bins=20, edgecolor='black')
    plt.title('Распределение чаевых')
    plt.xlabel('Чаевые ($)')
    plt.ylabel('Количество')
    
    # График 2: Зависимость чаевых от счета
    plt.subplot(1, 3, 2)
    plt.scatter(dataset['total_bill'], dataset['tip'], alpha=0.5)
    plt.title('Чаевые vs Счет')
    plt.xlabel('Счет ($)')
    plt.ylabel('Чаевые ($)')
    
    # График 3: Чаевые по категориям
    plt.subplot(1, 3, 3)
    dataset.boxplot(column='tip', by='day')
    plt.title('Чаевые по дням недели')
    plt.suptitle('')
    plt.xlabel('День')
    plt.ylabel('Чаевые ($)')
    
    plt.tight_layout()
    plt.show()
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")


if __name__ == "__main__":
    tips = sns.load_dataset("tips")
    main_info(tips)
    basic_statistics(tips)
    group_statistics(tips)
