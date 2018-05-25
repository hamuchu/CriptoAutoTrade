#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

import time
import datetime
import copy

from trade_class import TradeClass
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

with open('xgbmodel_btc.pickle', mode='rb') as f:
    model=pickle.load(f)

'''
model=xgb.XGBRegressor(learning_rate = 0.01, n_estimators=2000,
                           max_depth=10, min_child_weight=1,
                           gamma=0.2, objective= "reg:linear",
                           nthread=-1,scale_pos_weight=1, seed=27)
'''

#model.fit(X_train,y_train)


y_pred = model.predict(X_test)
predictions = [value for value in y_pred]
print("Predictions!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(predictions)

'''
#pickle.dump(model, open("xbgmodel.pickle", "wb"))
'''

with open('xgbmodel.pickle', mode='wb') as f:
   pickle.dump(model, f)

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
print(trade.simulate_trade(y_real_test,X_test,model))

'''
ZaifTrade
'''


from zaifapi import ZaifPublicApi
public_zaif = ZaifPublicApi()
import time
#添え字は0に近い方が強い。競っている
#入札をキャンセルする関数を作る。
# float(public_zaif.depth('btc_jpy')['asks'][0][0])#売
# float(public_zaif.depth('btc_jpy')['bids'][0][0])#買
print(public_zaif.depth('btc_jpy')['bids'][0][0])
from zaifapi import *
public_zaif = ZaifPublicApi()


trade_zaif = ZaifTradeApi('','')

history=[]
sleep_time=0.2
#askは売　bidは買
for num in range(3000):
    #time.sleep(60*5)#本当は5分×60セット待たなければならない
    ord_dict = trade_zaif.active_orders(currency_pair='btc_jpy')

    for key in ord_dict:
        print(key, ord_dict[key])
        trade_zaif.cancel_order(order_id=int(key))

    time.sleep(5)

    last_price = public_zaif.last_price('btc_jpy')[u'last_price']

    history.append(last_price)

    if len(history) <= 59:
        continue
    del history[0]

    prediction = model.predict(trade.TestPercentageLabel(history))#TODO:データの整形

    bids_top_price = public_zaif.depth('btc_jpy')['bids'][0][0]
    asks_top_price = public_zaif.depth('btc_jpy')['asks'][0][0]

    if prediction >0.0:
        trade_zaif.trade(currency_pair="btc_jpy", action="bid", price=int(bids_top_price+1), amount=0.0001)
    else:
        trade_zaif.trade(currency_pair="btc_jpy", action="ask", price=int(asks_top_price-1), amount=0.0001)
