#coding: utf-8
import requests
import json
#[UNIX timestamp, 始値, 高値, 安値, 終値, 出来高]
if __name__ == "__main__":
    #ローソク足の時間を指定
    #periods = ["60","300"]
    periods = ["600"]
    #クエリパラメータを指定
    query = {"periods":','.join(periods)}

    #ローソク足取得
    #１日:86400 1時間:3600　1分 60
    res = json.loads(requests.get("https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=600&after=1483196400",params=query).text)["result"]

    #表示
    print(res)
    f = open("HOUR-2017-6-1.json", "w")
    json.dump(res, f)
    '''
    for period in periods:
        print("period = "+period)
        row = res[period]
        length = len(row)
        for column in row:
            print (column)
    '''
