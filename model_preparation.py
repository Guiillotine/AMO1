import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

train_data = pd.read_csv('train/train_data_scaled.csv')

y_train = train_data['sales'].values
X_train = train_data.drop(['sales'], axis = 1)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model_fname = 'model.pkl'

joblib.dump(model, 'model.pkl')

print("Модель обучена ->", model_fname)