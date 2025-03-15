import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train/train_data.csv')
test_data = pd.read_csv('test/test_data.csv')

scaler = StandardScaler()

X_train = pd.DataFrame(train_data["sales"])
X_test = pd.DataFrame(test_data[["sales"]])

X_train_scaled = scaler.fit_transform(X_train).flatten()  # Обучение на train данных и их стандартизация
X_test_scaled = scaler.transform(X_test).flatten()

# Даты должны быть представлены в формате числа, чтобы использовать их как признак,
# поэтому преобразуем дату к номеру месяца и дня

def scaled_dateframe(scaled_sales, date_column):
    days = [pd.Timestamp(date).day for date in date_column.values]
    monthes = [pd.Timestamp(date).month for date in date_column.values]

    return pd.DataFrame({
        "month": monthes,
        "day": days,
        "sales": scaled_sales
    })

train_data_scaled = scaled_dateframe(X_train_scaled, train_data['date'])
test_data_scaled = scaled_dateframe(X_test_scaled, test_data['date'])

train_fname = 'train/train_data_scaled.csv'
test_fname = 'test/test_data_scaled.csv'

pd.DataFrame(train_data_scaled).to_csv('train/train_data_scaled.csv', index=False)
pd.DataFrame(test_data_scaled).to_csv('test/test_data_scaled.csv', index=False)

print("Данные стандартизованы ->", train_fname, test_fname)