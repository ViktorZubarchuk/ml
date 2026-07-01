import pandas as pd
import joblib
import numpy as np

# Загружаем модель
best_model = joblib.load('best_model.pkl')

def predict_price(brand, model, year, mileage_km, engine_volume, engine_hp, n_owners, transmission, body_type, color):    
    car = pd.DataFrame({
        'brand': brand, 
        'model': model,
        'year': year,
        'mileage_km': mileage_km,
        'engine_volume': engine_volume, 
        'engine_hp': engine_hp,
        'n_owners': n_owners,
        'transmission': transmission, 
        'body_type': body_type,
        'color': color
    }, index=[0])
    
    price = best_model.predict(car)[0]
    return price

if __name__ == "__main__":
    car = {
        'brand': 'BMW', 
        'model': '7 Series_M',
        'year': 2022,
        'mileage_km': 20000,
        'engine_volume': 3.0, 
        'engine_hp': 450,
        'n_owners': 2,
        'transmission': 'Automatic', 
        'body_type': 'Sedan',
        'color': 'black'
    }
    
    pred_price = predict_price(**car)

    print("\n\nПРЕДСКАЗАНИЕ ЦЕНЫ МАШИНЫ")
    print(pred_price)
