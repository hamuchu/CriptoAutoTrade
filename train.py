#coding: utf-8
#https://qiita.com/aaatsushi_bb/items/0b605c0f27493f005c88
import numpy
from trade_class import TradeClass
import xgboost as xgb
import copy
import numpy as np
from sklearn.grid_search import GridSearchCV
import matplotlib as plt
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

'''
with open('xgbmodel_btc.pickle', mode='rb') as f:
    model=pickle.load(f)
'''

'''
model=xgb.XGBRegressor(learning_rate = 0.01, n_estimators=2000,
                           max_depth=10, min_child_weight=1,
                           gamma=0.2, objective= "reg:linear",
                           nthread=-1,scale_pos_weight=1, seed=27)
'''


fit_params = {"early_stopping_rounds": 100,
              "eval_set": [[X_test, y_test]]}

xgb_model = xgb.XGBRegressor()
'''
params = {"learning_rate":[0.1,0.3,0.5],
        "max_depth": [2,3,5,10],
         "subsample":[0.5,0.8,0.9,1],
         "colsample_bytree": [0.5,1.0],
         }
gs = GridSearchCV(xgb_model,
                  params,
                  fit_params=fit_params,
                  cv=10,
                  n_jobs=-1,
                  verbose=2)

gs.fit(X_train, y_train)
'''
xgb_model.fit(X_train,y_train)

#model.fit(X_train,y_train)

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
print(trade.simulate_trade(y_real_test,X_test,model))
