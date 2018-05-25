# CriptoAutoTrade
仮想通貨の自動運用

## インストール
<ul>
<li>conda install でxgboostをインストールすると上手くいく</li>
<li>pip install xgboostだと上手く行かないから、Anaconda cloudを使用する。</li>
<li>pickleファイルは、モデルを保存したファイル</li>
<li>xgboostはscikit-learnと同じ使い方ができるが、異なるライブラリである。ハイパーパラメータサーチをすることもできる。</li>
</ul>


##　TODO
<ul>
<li>1.　取引所の口座を開設</li>
<li>2.　過去のヒストリカルデータを取得(http://www.bitbityen.com/entry/2017/11/22/080000)</li>
<li>3. 　XGBoostやディープラーニングで学習させる</li>
<li>4. 　注文を出すロジックを組む。（指値や刺さらなかった注文のキャンセルなど）<li>
<li>5.　実運用</li>

<br>
2~5までを繰り返し行うことによって、利益率を上げていく。<br/>
bitflyerからもヒストリカルデータを取得することができる。<br/>
自然言語処理の部分は学習段階と実運用の段階で必要。</br>
強化学習も取り入れて板読みトレードもさせたい。<br/>

