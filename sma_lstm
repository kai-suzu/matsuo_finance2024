import numpy as np
import pandas as pd

import sys
import os
from typing import NamedTuple, Optional
from dataclasses import dataclass

from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import optuna
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import warnings
import pandas_ta as ta
import copy
from mlbacktester import Order, BaseStrategy, AssetInfo
from mlbacktester.bt import BackTester

def make_second_prediction(df: pd.DataFrame):

    df_ = df.copy()

    # リターン率の計算 (1時間後のclose価格の変動率)
    df_['ret'] = df_.groupby(level=1)['close'].pct_change(periods=-1)  #groupbyでマルチカラムに対応

    # メタラベルの計算
    df_['metalabel'] = np.where(df_['ret'] * df_['side'] > 0, 1, 0)

    # NaNを含む行を削除
    df_new = df_.dropna()

    return df_new

def create_sequences(df, sequence_length=48):
    X, y_metalabel, y_ret = [], [], []
    scaler = StandardScaler()

    # 1行ずつずらしながらシーケンスを生成
    for i in range(len(df) - sequence_length + 1):
        sequence = df.iloc[i:i+sequence_length]
        scaled_sequence = scaler.fit_transform(sequence[['open', 'high', 'low', 'close', 'volume', 'side']])
        X.append(scaled_sequence)
        y_metalabel.append(sequence.iloc[-1]['metalabel'])  # 最後の行のmetalabelを取得
        y_ret.append(sequence.iloc[-1]['ret'])  # 最後の行のretを取得

    return np.array(X), np.array(y_metalabel), np.array(y_ret)

def objective(trial, X_train, y_train_metalabel, y_train_ret, X_val, y_val_metalabel, y_val_ret):
    # ハイパーパラメータの提案（units, dropout_rate, learning_rateのみ変更可能に）
    # レイヤー数を2に固定
    n_layers = 2
    units = trial.suggest_int('units', 50, 150)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # LSTMモデルの構築

    # 入力層の定義
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    # 最初のLSTM層
    x = LSTM(units=units, return_sequences=(n_layers > 1))(inputs)
    x = Dropout(dropout_rate)(x)
    # 2番目のLSTM層（n_layersが2なので、ここでreturn_sequencesはFalseに）
    x = LSTM(units=units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    # メタラベルのための出力層
    output1 = Dense(1, activation='sigmoid', name='meta_label_output')(x)
    # リターンの予測のための出力層
    output2 = Dense(1, name='return_output')(x)
    # モデルの定義
    model = Model(inputs=inputs, outputs=[output1, output2])
    # モデルのコンパイル
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=['binary_crossentropy', 'mse'])

    # ターゲット変数をリストとして明示的に定義し、model.fitに渡す
    model.fit(X_train, [y_train_metalabel, y_train_ret],
              validation_data=(X_val, [y_val_metalabel, y_val_ret]),
              epochs=10, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    # 予測値の取得と処理
    predictions = model.predict(X_val)
    pred_metalabel = predictions[0].flatten()   # メタラベル予測
    pred_ret = predictions[1].flatten()       # リターン予測

    # 損失の計算
    loss_metalabel = log_loss(y_val_metalabel, pred_metalabel)
    loss_ret = mean_squared_error(y_val_ret, pred_ret)

    # 合計損失を返す
    total_loss = loss_metalabel + loss_ret
    return total_loss

def walk_forward_optimization(X, y_metalabel, y_ret, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_loss = float('inf')
    best_params = None

    for train_index, test_index in tscv.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train_metalabel, y_val_metalabel = y_metalabel[train_index], y_metalabel[test_index]
        y_train_ret, y_val_ret = y_ret[train_index], y_ret[test_index]

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda trial: objective(trial, X_train, y_train_metalabel, y_train_ret, X_val, y_val_metalabel, y_val_ret), n_trials=5, show_progress_bar=True)

    if study.best_trial.value < best_loss:
        best_loss = study.best_trial.value
        best_params = study.best_trial.params

    print(f"Best trial for split {train_index[-1]}-{test_index[-1]}:")
    print(study.best_trial.params)

    return best_params

def make_model(X, y_metalabel, y_ret, best_params, n_splits=5):
    # ベストパラメータの展開
    units = best_params['units']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']

    # TimeSeriesSplitの定義
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # TimeSeriesSplitを用いた訓練データと検証データの分割
    # ここでは、最後の分割のみを使用してモデルを訓練します。
    # ループが終了した時点でのX_train, X_val, y_train_metalabel, y_val_metalabel, y_train_ret, y_val_retが訓練に使用されます。
    for train_index, test_index in tscv.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train_metalabel, y_val_metalabel = y_metalabel[train_index], y_metalabel[test_index]
        y_train_ret, y_val_ret = y_ret[train_index], y_ret[test_index]

    # モデルの構築
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(units=units, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    output1 = Dense(1, activation='sigmoid', name='meta_label_output')(x)
    output2 = Dense(1, name='return_output')(x)
    model = Model(inputs=inputs, outputs=[output1, output2])

    # モデルのコンパイル
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=['binary_crossentropy', 'mse'])

    # モデルの訓練
    model.fit(X_train, [y_train_metalabel, y_train_ret],
              validation_data=(X_val, [y_val_metalabel, y_val_ret]),
              epochs=10, batch_size=32, verbose=0)

    # 訓練済みモデルの返却
    return model

def create_sequences_pred(df, sequence_length=48):
    X = []
    scaler = StandardScaler()

    # データフレームの長さまで1行ずつずらしながらループ
    for i in range(len(df) - sequence_length + 1):
        sequence = df.iloc[i:i + sequence_length]
        scaled_sequence = scaler.fit_transform(sequence[['open', 'high', 'low', 'close', 'volume', 'side']])
        X.append(scaled_sequence)

    return np.array(X)


class Strategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
    
    def preprocess(self, df):

        df_ = df.copy()
        df_new_list = []

        for symbol in df_.index.levels[1].unique():
            df_symbol = df_.loc[(slice(None), symbol), :]

            sma5 = ta.sma(df_symbol['close'], length=5)  #単純移動平均
            sma25 = ta.sma(df_symbol['close'], length=25)

            # ゴールデンクロス, デッドクロスによる予測を導出
            diff = sma5 - sma25
            df_symbol['buy'] = np.where((np.sign(diff) - np.sign(diff.shift(1)) == 2), 1, 0) # ゴールデンクロス
            df_symbol['sell'] = np.where((np.sign(diff) - np.sign(diff.shift(1)) == -2), -1, 0) # デッドクロス
            df_symbol['side'] = df_symbol['buy'].values + df_symbol['sell'].values

            df_new_list.append(df_symbol)

        df_new = pd.concat(df_new_list).sort_index()

        return df_new

    def get_model(self, train_df):
        train_df = make_second_prediction(train_df)
        symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']  # シンボルリスト
        models = []
        for symbol in symbols:
            train_df_symbol = train_df.loc[(slice(None), symbol), :]
            X, y_metalabel, y_ret = create_sequences(train_df_symbol, sequence_length=48)
            best_params = walk_forward_optimization(X, y_metalabel, y_ret, n_splits=5)
            model = make_model(X, y_metalabel, y_ret, best_params, n_splits=5)

            models.append(model)

        return models

    def get_signal(self, preprocessed_df: pd.DataFrame, models: list = None):

        sequence_length = 48
        df_ = preprocessed_df.copy()

        # シンボルのリスト
        symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']

        # 各シンボルに対応したモデルを用いて予測を行い、結果を追加する
        for symbol, model in zip(symbols, models):
            # シンボルに対応するデータのみを選択
            df_symbol = df_.loc[(slice(None), symbol), :]

            #データのシーケンス化
            df_symbol = create_sequences_pred(df_symbol)

            # ここでモデルを用いて予測を行う
            prediction = model.predict(df_symbol)
            pred_meta = prediction[0]  # メタラベルの予測結果
            pred_ret = prediction[1]  # リターンの予測結果

            # 予測結果を取得
            pred_meta = np.concatenate([np.full((sequence_length-1,), np.nan), prediction[0].flatten()])
            pred_ret = np.concatenate([np.full((sequence_length-1,), np.nan), prediction[1].flatten()])

            # 元のデータフレームに予測結果を追加
            df_.loc[(slice(None), symbol), 'pred_meta'] = pred_meta
            df_.loc[(slice(None), symbol), 'pred_ret'] = pred_ret

        # 条件に基づいてorder_flag列を作成
        df_['order_flag'] = (
        ((df_['side'] == 1) & (df_['pred_meta'] > 0.5) & (df_['pred_ret'] > 0)) |
        ((df_['side'] == -1) & (df_['pred_meta'] > 0.5) & (df_['pred_ret'] < 0))
        ).astype(int)

        return df_    

    def get_orders(
        self,
        latest_timestamp: pd.Timestamp,
        latest_bar: pd.DataFrame,
        latest_signal: pd.DataFrame,
        asset_info: AssetInfo,
        ) -> list[Order]:

        order_lst = []
        symbols = latest_signal.index.get_level_values(1).unique()

        for symbol in symbols:
            pos_size = asset_info.pos_size[symbol]
            avg_price = asset_info.avg_price[symbol]
            # 注文を出すsymbolの情報を取得する
            latest_signal_symbol = latest_signal.loc[(latest_timestamp, symbol), :]

            # 最大ポジションを設定し、現在のポジションがそれを超えてないかチェックする
            lot_size_dollar = asset_info.nav / 20
            max_lot =  asset_info.nav / 10 #symbolの現在の総資産
            position_price = pos_size * avg_price
            max_pos_flag = position_price > max_lot
            # 利確と損切のフラグを作成する
            pt_lc_thre = 0.005
            pt_flag = latest_signal_symbol["close"] > avg_price * (1 + pt_lc_thre)
            lc_flag = latest_signal_symbol["close"] < avg_price * (1 - pt_lc_thre)

            if (pos_size != 0) and (max_pos_flag or pt_flag or lc_flag):
                # 最大ポジションまたは利確損切ラインに達しているとき
                # ポジションを解消する
                if pos_size > 0:
                    order_lst.append(Order(type="MARKET",
                                           side="SELL",
                                           size=abs(pos_size),
                                           price=None,
                                           symbol=symbol))
                elif pos_size < 0:
                    order_lst.append(Order(type="MARKET",
                                           side="BUY",
                                           size=abs(pos_size),
                                           price=None,
                                           symbol=symbol))

            # シグナルに従って発注を行う
            else:
                if latest_signal_symbol["order_flag"] == 1:
                    order_size = lot_size_dollar / latest_signal_symbol["close"]
                    order_size = int(order_size / 0.1) * 0.1
                    side = "BUY" if latest_signal_symbol["side"] == 1 else "SELL"
                    order_lst.append(Order(type="MARKET",
                                           side=side,
                                           size=order_size,
                                           price=None,
                                           symbol=symbol))
        return order_lst
