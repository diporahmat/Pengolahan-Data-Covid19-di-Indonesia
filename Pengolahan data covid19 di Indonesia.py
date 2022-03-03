import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('covid_indonesia_2.csv')
print(data.head())

print(data.describe())

print(data.dtypes)

data['cases_norm'] = np.log(data[['Kasus_harian']])
print(data)

# mengubah kolom tanggal menjadi kolom numerik dengan 1 Juli 2021 sebagai dasar
default_date = pd.to_datetime(data['Tanggal'])[0]
print(f'nilai default_date adalah : {default_date}')

# masukkan pada sebuah kolom baru
data['days'] = data['Tanggal'].apply(lambda x: pd.Timedelta(pd.to_datetime(x) - default_date).days)
print(data)

df_covid = data[['cases_norm','days']]
print(df_covid)

# pembuatan model
x = df_covid['days']
y = df_covid['cases_norm']

x = np.array(x).reshape((-1, 1))
y = np.array(y).reshape((-1, 1))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(f'dimensi dari X_train : {len(X_train)}')
print(f'dimensi dari y_train : {len(y_train)}')
print(f'dimensi dari X_test : {len(X_test)}')
print(f'dimensi dari y_test : {len(y_test)}')

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)

poly_x_test = poly_reg.fit_transform(X_test)

y_pred = lin_reg.predict(poly_x_test)

# evaluasi model
r2 = metrics.r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('R2 score is {}'.format(r2))

# membuat predict date
predict_date = np.array([[137],
       [138],
       [139],
       [140],
       [141],
       [142],
       [143],
       [144],
       [145],
       [146],
       [147],
       [148],
       [149],
       [150],
       [151],
       [152],
       [153],
       [154],
       [155],
       [156],
       [157],
       [158],
       [159],
       [160],
       [161],
       [162],
       [163],
       [164],
       [165],
       [166]])

date_x = poly_reg.fit_transform(predict_date)
predict_cases = lin_reg.predict(date_x)
print(predict_cases)

# visualisasi
plt.scatter(predict_date, predict_cases)
plt.title('Forecasting kasus Baru Covid di Indonesia')
plt.xlabel('X-Hari Setelah 1 Juli 2021')
plt.ylabel('Jumlah Kasus Baru')
plt.show()
