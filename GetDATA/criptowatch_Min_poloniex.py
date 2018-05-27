#coding: utf-8
import requests
import json
#[UNIX timestamp, 始値, 高値, 安値, 終値, 出来高]
#http://url-c.com/tc/
#2018/05/23 12:02:00 2018/05/23 12:03:00
if __name__ == "__main__":
    #ローソク足の時間を指定
    #periods = ["60","300"]
    periods = ["60"]
    #クエリパラメータを指定
    query = {"periods":','.join(periods)}

    #ローソク足取得
    #１日:86400 1時間:3600　1分 60
    res = json.loads(requests.get("https://api.cryptowat.ch/markets/poloniex/btcusdt/ohlc?periods=60&after=1483196400",params=query).text)["result"]

    #表示
    print(res)
    f = open("Min-poloniex.json", "w")
    json.dump(res, f)
    '''
    for period in periods:
        print("period = "+period)
        row = res[period]
        length = len(row)
        for column in row:
            print (column)
    '''
