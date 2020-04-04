# kaggle-m5-accuracy-library
kaggleウォルマートm5の自作ライブラリ用のレポジトリです。
※以下、dfはpandasデータフレーム、ndarrayはnumpy配列を指す


#change_wm_yr_wk_to_weeknum.change_wm_yr_wk()
引数(1):df
戻り値:df

詳細:引数に指定したデータフレームから"wm_yr_wk"のカラムの列を抽出し、
     2011年1月第4週目を1として通し番号を付けていく。
     
     
#reduce_mem_usage.reduce_mem_usage
引数(1):df
戻り値:df

詳細:引数に指定したデータフレームの最適な型(int8,int16,int32...)を見つけ、自動で変換する。
     メモリ削減につながる。
     
     
#metari.calculate_WRMSSE()
引数(6):df,df,df,int,int,ndarray
戻り値:float

詳細:M5の精度指標である加重(重みw)と2乗平均平方誤差(RMSSE)を乗算した値(WRMSSE)を求める関数
ここで言う重みは「all,state,state-store,cat,....item」と細かく求められ、その値は売り上げ個数や
レベルによって変わる。最終的に42840*42840のarray同時の計算(要素の掛け合わせ)を行い結果を出す。
ある商品について、初めて商品が売れた時から誤差の計算が開始される(期間中1度も売れなかった場合、
予測値に依存せず誤差は0)

具体的な使用方法:
第一引数=sales_train_validation.csvから読んだdf,第二引数=calender.csvから呼んだdf,
第三引数=sell_prices.csvから呼んだdf,第四引数=訓練、テストを合わせた初めのd(日付)
第五引数=訓練、テストを合わせた終わりののd(日付),第六引数=テストの予測値

※予測値の並び順は

HOBBIES_1_001_CA_1_validationのday1の予測データ
-----------------------------------------------
HOBBIES_1_001_CA_1_validationのday2の予測データ
-----------------------------------------------
...
のように、元のid順を変えないままそれぞれのidで時系列順にする

使用例:
y_pred = model.predict(test)
train_df = pd.read_csv("sales_train_validation.csv")
calender = pd.read_csv("calendar.csv")
prices = pd.read_csv("sell_prices.csv")
metari.calculate_WRMSSE(train_df,calender,prices,1,365,y_pred)
...実行
score:hoge

※まあまあ時間がかかります。

