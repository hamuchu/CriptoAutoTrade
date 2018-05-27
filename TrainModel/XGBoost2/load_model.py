#coding: utf-8
#https://qiita.com/aaatsushi_bb/items/0b605c0f27493f005c88
import numpy
from trade_class import TradeClass
import xgboost as xgb
import copy
import numpy as np
from sklearn.grid_search import GridSearchCV
import matplotlib as plt
import matplotlib.pyplot as plt

trade=TradeClass()
time_date, price_data = trade.getDataPoloniex()

max_price=max(price_data)
training_set=copy.copy(price_data)


X_train = []
y_train = []
for i in range(60, len(training_set)-10001):
    #X_train.append(np.flipud(training_set_scaled[i-60:i]))
    X_train.append(training_set[i - 60:i])
    y_train.append(training_set[i])

X_train,y_train=trade.PercentageLabel(X_train,y_train)

X_train, y_train = np.array(X_train,dtype='float64'), np.array(y_train,dtype='float64')

from scipy.sparse import coo_matrix
X_sparse=coo_matrix(X_train)

import sklearn
X,X_sparse,y = sklearn.utils.shuffle(X_train,X_sparse,y_train,random_state=0)

X_test = []
y_test = []
for i in range(len(training_set)-10000, len(training_set)):
    X_test.append(training_set[i - 60:i])
    y_test.append(training_set[i])
X_test, y_test = np.array(X_test,dtype='float64'), np.array(y_test,dtype='float64')

#PercentageLabelで配列の長さが１つ分短くなっている
y_real_test=y_test#[1:]

X_test,y_test = trade.PercentageLabel(X_test,y_test)

import xgboost as xgb
import pickle


with open('xgbmodel.pickle', mode='rb') as f:
    xgb_model=pickle.load(f)

y_pred = xgb_model.predict(X_test)
predictions = [value for value in y_pred]
print("Predictions!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(predictions)


pickle.dump(xgb_model, open("xbgmodel.pickle", "wb"))

'''
with open('xgbmodel.pickle', mode='wb') as f:
   pickle.dump(model, f)
'''

#loaded_model = pickle.load(open("xgbmodel.pickle.dat", "rb"))
plt.plot(y_test, color = 'red', label = 'percentage_teacher')
plt.plot(predictions, color = 'blue', label = 'percentage')
plt.plot(y_real_test, color = 'green', label = 'Real Price')
plt.title('Cripto currency prediction')
plt.xlabel('Time')
plt.ylabel('green:Real Price blue: prediction(%) red: teacher(%)')
plt.legend()
plt.show()

print("FINAL money")
print(trade.simulate_trade(y_real_test,X_test,xgb_model))
