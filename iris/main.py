import subprocess
import os

def run_project():
    # 1. EDA
    subprocess.run(['python', 'iris/01_eda.py'])
    
    # 2. Подбор модели
    subprocess.run(['python', 'iris/02_model_selection.py'])
    
    # 3. Предсказание
    subprocess.run(['python', 'iris/03_predict.py'])
    

if __name__ == "__main__":
    run_project()