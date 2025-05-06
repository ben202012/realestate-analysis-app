# models.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

def train_price_prediction_model(df, features):
    """価格予測モデルを構築"""
    if 'TradePrice' not in df.columns:
        st.error('価格データが存在しません')
        return None, None, None
    
    # 特徴量が全てDataFrameに存在するか確認
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"以下の特徴量がデータに存在しません: {', '.join(missing_features)}")
        return None, None, None
    
    # NaNを含む行を除外
    model_df = df.dropna(subset=features + ['TradePrice'])
    
    if len(model_df) < 20:  # 訓練に必要な最小サンプル数
        st.error(f"モデル構築に必要なデータが不足しています。現在のサンプル数: {len(model_df)}")
        return None, None, None
    
    # 特徴量と目的変数の準備
    X = model_df[features]
    y = model_df['TradePrice']
    
    # トレーニングデータとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # モデルの初期化と訓練
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # テストデータでの予測
    y_pred = model.predict(X_test)
    
    # モデル評価
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

def show_feature_importance(model, feature_names):
    """特徴量の重要度を表示"""
    if model is None:
        return
    
    # 特徴量の重要度を取得
    importances = model.feature_importances_
    
    # 重要度をDataFrameに変換
    importance_df = pd.DataFrame({
        '特徴量': feature_names,
        '重要度': importances
    }).sort_values('重要度', ascending=False)
    
    # 可視化
    fig = px.bar(
        importance_df,
        x='特徴量',
        y='重要度',
        title='特徴量の重要度',
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return importance_df

def predict_price(model, input_features, feature_names):
    """ユーザー入力値から価格を予測"""
    if model is None:
        return None
    
    # 入力値をDataFrameに変換
    input_df = pd.DataFrame([input_features], columns=feature_names)
    
    # 予測
    prediction = model.predict(input_df)
    
    return prediction[0]