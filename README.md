# kaggle-m5-accuracy-library
kaggleウォルマートm5の自作ライブラリ用のレポジトリです。

#change_wm_yr_wk()
引数:pandasデータフレーム
戻り値:pandasデータフレーム
詳細:引数に指定したデータフレームから"wm_yr_wk"のカラムの列を抽出し、
     2011年1月第4週目を1として通し番号を付けていく。
     
     
#reduce_mem_usage()
引数:pandasデータフレーム
戻り値:pandasデータフレーム
詳細:引数に指定したデータフレームの最適な型(int8,int16,int32...)を見つけ、自動で変換する。
     メモリ削減につながる。
