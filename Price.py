import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from datetime import datetime


dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            # date_str = row[0]
            # date_obj = str(datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S'))
            dates.append(int(row[0].split('/')[0]))
            prices.append(float(row[1]))
    return

def predict_price(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    #using support vector regression to plot and pridict the next value
    SVR_lin = SVR(kernel= 'linear', C=1e3)
    SVR_poly = SVR(kernel='poly', C=1e3, degree= 2)
    SVR_rbf = SVR(kernel='rbf' , C=1e3, gamma=0.1)
    SVR_lin.fit(dates, prices)
    SVR_poly.fit(dates, prices)
    SVR_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, SVR_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, SVR_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, SVR_poly.predict(dates), color='blue', label = 'Polynomial mode')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('support Vector regression')
    plt.legend()
    plt.show()

    return SVR_rbf.predict(x)[0], SVR_lin.predict(x)[0], SVR_poly.predict(x)[0]

get_data('aapl.csv')

predicted_price = predict_price(dates, prices, 29)

print(predicted_price)