import numpy as np
import pandas as pd
import os

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

DAYS_AT_YEAR = 365
monthes_names = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthes = {name: num+1 for num, name in enumerate(monthes_names)}


def generate_data(years_num: int, noise: int, base_sales: int = 100):
    """Генерация данных об изменении спроса на кондиционеры для сети магазинов (покупок в день)"""
    sales_data = []
    init_date = pd.Timestamp("2025-01-01")
    cur_date = init_date

    while cur_date.year < init_date.year+years_num:
        cur_month_num = cur_date.month

        day_of_year = (cur_date - pd.Timestamp(f"{cur_date.year}-01-01")).days
        amplitude = 10
        season_effect = amplitude * np.sin(2 * np.pi * cur_date.day / DAYS_AT_YEAR) # Эффект сезона
        base_sales += season_effect

        changed_on = np.random.normal(0, noise) # Шум

        # Летний сезон
        if cur_month_num >= monthes["June"] and cur_month_num <= monthes["Aug"]:
            if cur_month_num == monthes["July"]:
                changed_on = np.random.randint(base_sales * 5, base_sales * 6)
            else:
                changed_on = np.random.randint(base_sales * 4,base_sales * 5)

        # Зимний сезон
        elif cur_month_num in [monthes["Dec"], monthes["Jan"], monthes["Feb"]]:
            changed_on = -np.random.randint(base_sales // 1.1, base_sales)

        # Остальные месяцы
        else:
            changed_on = -np.random.randint(base_sales // 3.5, base_sales // 3)

        sales_data.append({
            "date": cur_date,
            "sales": int(base_sales + changed_on)
        })

        cur_date += pd.Timedelta(days=1)

    return pd.DataFrame(sales_data)


train_data = generate_data(years_num=3, noise=5)
test_data = generate_data(years_num=3, noise=10)

train_fname = 'train/train_data.csv'
test_fname = 'test/test_data.csv'

pd.DataFrame(train_data).to_csv(train_fname, index=False)
pd.DataFrame(test_data).to_csv(test_fname, index=False)

print("Данные сгенерированы ->", train_fname, test_fname)