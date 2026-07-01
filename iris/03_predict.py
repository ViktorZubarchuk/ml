import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Загружаем модель
model = joblib.load('iris/best_model.pkl')

# Загружаем iris для названий классов
iris = load_iris()


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    new_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

    pred = model.predict(new_data)[0]
    return pred


if __name__ == "__main__":
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    pred_class = predict_iris(**data)

    print("\n🌸 ПРЕДСКАЗАНИЕ ИРИСА")
    print("Класс (число):", pred_class)
    name = iris.target_names[pred_class]
    print(f"Класс: {pred_class} ({name})")