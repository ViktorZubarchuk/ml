import pandas as pd
import joblib
import numpy as np

# Загружаем модель
model = joblib.load('tips/best_model.pkl')

def predict_tip(total_bill, size, sex, smoker, day, time):    
    new_order = pd.DataFrame({
        'total_bill': [total_bill],
        'size': [size],
        'sex': [sex],
        'smoker': [smoker],
        'day': [day],
        'time': [time]
    })
    
    tip = model.predict(new_order)[0]
    return tip

if __name__ == "__main__":
    order = {
        'total_bill': 50.50,
        'size': 10,
        'sex': 'Male',
        'smoker': 'Yes',
        'day': 'Sun',
        'time': 'Dinner'
    }
    
    predicted_tip = predict_tip(**order)
    tip_percent = (predicted_tip / order['total_bill']) * 100
    
    print("\n\nПРЕДСКАЗАНИЕ ЧАЕВЫХ")
    print(f"Счет:          ${order['total_bill']:.2f}")
    print(f"Количество:    {order['size']} чел.")
    print(f"Пол:           {order['sex']}")
    print(f"Курит:         {order['smoker']}")
    print(f"День:          {order['day']}")
    print(f"Время:         {order['time']}")
    print(f"\nРекомендуемые чаевые: ${predicted_tip:.2f}")
    print(f"Процент чаевых:       {tip_percent:.1f}%")