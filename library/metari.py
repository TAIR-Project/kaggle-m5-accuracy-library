#!/usr/bin/env python
# coding: utf-8

# In[6]:


from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import warnings
class WRMSSEEvaluator(object):
    
    group_ids = ( 'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],
        ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])

    def __init__(self, 
                 train_df: pd.DataFrame, 
                 valid_df: pd.DataFrame, 
                 calendar: pd.DataFrame, 
                 prices: pd.DataFrame):
        '''
        intialize and calculate weights
        '''
        warnings.simplefilter('ignore', RuntimeWarning)
        print("重みとRMSSEを計算中...")
        
        
        self.calendar = calendar
        self.prices = prices
        self.train_df = train_df
        self.valid_df = valid_df
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]
        
        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()
        self.train_df['all_id'] = "all"

        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]

        #self.dfにid部分をくっつける
        #if not all([c in self.valid_df.columns for c in self.id_columns]):
        self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df],
                                 axis=1, 
                                sort=False)
        
        self.train_series = self.trans_30490_to_42840(self.train_df, 
                                                      self.train_target_columns, 
                                                      self.group_ids)
        #{"all":{d1総計,d2総計.....}, "CA":{d1総計,d2総計....}, .......}
        self.valid_series = self.trans_30490_to_42840(self.valid_df, 
                                                      self.valid_target_columns, 
                                                      self.group_ids)
        #{"all":{d-28総計,d-27総計.....}, "CA":{d-28総計,d-27総計....}, .......}
        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        
        self.train_series = None
        self.train_df = None
        self.prices = None
        self.calendar = None

    def get_scale(self):
        '''
        scaling factor for each series ignoring starting zeros
        '''
        scales = []
        for i in tqdm(range(len(self.train_series))):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series!=0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        #全区間が0のものはscale=NaN
        #scale = ((series[1:] - series[:-1]) ** 2).mean()が0のものがrmsse=infになる
        return np.array(scales)
    
    def get_name(self, i):
        '''
        convert a str or list of strings to unique string 
        used for naming each of 42840 series
        '''
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)
    
    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        #calenderからd_1,d_2をindexとしたwm_yr_wkのSeriesを作りto_dict()でmapに変換
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        
        #item_id,store_idとd-28～のみのリストを作る
        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(
            ["item_id", "store_id"]
        )
        
        
        weight_df = (
            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        )
       
        #d_-28～のデータをwm_yr_wkのmapで上書き
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        
        #priceデータと結合
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        
        #item_id,store_id,売上(value)のdfにする
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
            "value"
        ]
        
        weight_df = weight_df.loc[
            zip(self.train_df.item_id, self.train_df.store_id), :
        ].reset_index(drop=True)
        weight_df = pd.concat(
            [self.train_df[self.id_columns], weight_df], axis=1, sort=False
        )
        weights_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False)):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array(
                    [lv_weight.iloc[i]]
                )
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):
        '''
        transform 30490 series to all 42840 series
        '''
        series_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):
            tr = df.groupby(group_id)[cols].sum()
            #1:all_idについてd_1の総計,d2の総計...を出す→shape(1,about1900)
            #3:state_idについてd_1の総計,d2の総計...を出す
            for j in range(len(tr)):
                series_map[self.get_name(tr.index[j])] = tr.iloc[j].values
                #2:all_idについて集計したd1の総計...をarrayとしてmapに追加(キーは)
        return pd.DataFrame(series_map).T
    
    def get_rmsse(self, valid_preds) -> pd.Series:
        '''
        returns rmsse scores for all 42840 series
        '''
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (score / self.scale).map(np.sqrt)
        #scaleが0ならrmsseはinfになる(考慮しなくてよい)
        #scaleがinfならrmsseは0になる(考慮しなくてよい)
        return rmsse

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape
        
        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds],
                                axis=1, 
                                sort=False)
        valid_preds = self.trans_30490_to_42840(valid_preds, 
                                                self.valid_target_columns, 
                                                self.group_ids, 
                                                True)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse], 
                                      axis=1, 
                                      sort=False).prod(axis=1)
        self.contributors.drop(self.contributors[np.isinf(self.contributors.values)].index,inplace=True)
        return np.sum(self.contributors)


# In[7]:


#input
def calculate_WRMSSE(train_df,calendar,prices,start_d,end_d,y_pred):
    print("精度計算の前の前準備を実行中...")
    start = 'd_' + str(start_d)
    end = 'd_' + str(end_d)
    answear = train_df.drop(["item_id","dept_id","cat_id","store_id","state_id"],axis=1)
    answear = pd.melt(answear ,id_vars=answear.columns[0], var_name='date', value_name='sell_num')
    answear["date"] = answear["date"].apply(lambda x:x[2:])
    answear["date"] = answear["date"].astype('int16')
    answear = answear.loc[(answear["date"] > int(end[2:]) - 28)&(answear["date"] <= int(end[2:]))]
    answear.reset_index(inplace=True)
    answear.drop("index",axis=1,inplace=True)
    keys, i = np.unique(answear["id"], return_index=True)
    keys = answear["id"][i[i.argsort()]].values
    d = dict(zip(keys, np.arange(len(keys))))
    answear["category"] = answear["id"].map(d)
    answear.sort_values(["category","date"],inplace=True)
    answear.reset_index(inplace=True)
    answear.drop(["index","category"],axis=1,inplace=True)
    answear.loc[:,'sell_num']=y_pred
    answear["Unnamed: 0"] = answear.index
    answear["date"] = answear.groupby("id")["Unnamed: 0"].transform(lambda x:338 + (x % 28))
    keys, i = np.unique(answear["id"], return_index=True)
    keys = answear["id"][i[i.argsort()]].values
    d = dict(zip(keys, np.arange(len(keys))))
    answear["category"] = answear["id"].map(d)
    df_pivot = pd.pivot_table(answear, index=['category'], columns = ['date'],values=['sell_num'])
    drop_columns = list(train_df.columns[6:6+int(start[2:])-1])
    drop_columns.extend(train_df.columns[6+int(end[2:]):])
    train_df.drop(train_df.columns[371:],axis=1,inplace=True)
    train_fold_df = train_df.iloc[:, :-28]
    valid_fold_df = train_df.iloc[:, -28:].copy()
    e = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
    del train_fold_df, train_df, calendar, prices
    print("精度を計算中...")
    wrmsse = e.score(df_pivot.values)
    del df_pivot, answear
    return wrmsse

