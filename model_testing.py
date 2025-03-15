import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error


model = joblib.load('model.pkl')
test_data = pd.read_csv('test/test_data_scaled.csv')

#X_test = pd.DataFrame(test_data['date'])
y_test = test_data['sales'].values
X_test = test_data.drop(['sales'], axis = 1)

y_predict = model.predict(X_test)

# Оценка
mse = mean_squared_error(y_test, y_predict)

def check_mse_for_standarted_data(mse: float):
    """Проверка MSE для стандартизованных данных"""
    if mse < 0.9:
        print("Хорошее качество модели")
    elif mse < 1:
        print("Случайное угадывание")
    else:
        print("Плохое качество модели")


print(f"Среднеквадратичное отклонение: {mse}")
check_mse_for_standarted_data(mse)