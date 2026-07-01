import joblib
import pandas as pd
import numpy as np

model = joblib.load('best_model.pkl')

def predict_price(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    house = {
        'MedInc':
    }