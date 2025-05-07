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
    
    # カテゴリカル変数の処理
    categorical_features = ['Type', 'Renovation']
    categorical_features_present = [f for f in categorical_features if f in features]
    
    # 元の特徴量を保存
    st.session_state['original_features'] = features
    
    # 各カテゴリの選択肢を保存
    for cat_feature in categorical_features_present:
        available_values = model_df[cat_feature].dropna().unique().tolist()
        st.session_state[f'available_{cat_feature.lower()}s'] = available_values
    
    if categorical_features_present:
        # One-Hot Encoding
        model_df = pd.get_dummies(model_df, columns=categorical_features_present, drop_first=False)
        
        # ダミー変数化された特徴量を取得
        dummy_features = []
        for f in categorical_features_present:
            dummy_cols = [col for col in model_df.columns if col.startswith(f'{f}_')]
            dummy_features.extend(dummy_cols)
        
        # 元の特徴量リストからカテゴリカル変数を削除し、ダミー変数を追加
        features_numeric = [f for f in features if f not in categorical_features_present]
        model_features = features_numeric + dummy_features
    else:
        model_features = features
    
    # 特徴量と目的変数の準備
    X = model_df[model_features]
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
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # モデルと特徴量をセッション状態に保存
    st.session_state['prediction_model'] = model
    st.session_state['model_features'] = model_features
    st.session_state['categorical_features'] = categorical_features_present
    st.session_state['model_r2'] = r2
    st.session_state['model_mse'] = mse
    st.session_state['model_rmse'] = rmse  # RMSEも保存
    
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

# models.py に追加する関数

def predict_with_model(input_values=None):
    """セッションに保存されたモデルを使用して予測を実行"""
    # モデルと特徴量がセッションにあるか確認
    if 'prediction_model' not in st.session_state or 'model_features' not in st.session_state:
        st.error("先にモデルを構築してください")
        return None
    
    model = st.session_state['prediction_model']
    features = st.session_state['model_features']
    categorical_features = st.session_state.get('categorical_features', [])
    
    # 入力値をDataFrameに変換
    if input_values is not None:
        input_df = pd.DataFrame([input_values])
    else:
        st.error("入力値がありません")
        return None
    
    # カテゴリカル変数のOne-Hot Encoding
    if categorical_features:
        input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=False)
        
        # モデルが必要とする特徴量と一致させる
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # 不足している列を0で埋める
    
    # モデルが使用する特徴量のみを選択
    try:
        input_df = input_df[features]
    except KeyError as e:
        st.error(f"特徴量の不整合: {str(e)}")
        st.write("モデル特徴量:", features)
        st.write("入力特徴量:", input_df.columns.tolist())
        return None
    
    # 予測実行
    try:
        prediction = model.predict(input_df)[0]
        # 予測結果をセッションに保存
        st.session_state['prediction_result'] = prediction
        return prediction
    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {str(e)}")
        return None

def show_prediction_form_and_results(df):
    """予測フォームと結果を表示する関数"""
    # モデルが構築済みか確認
    if 'prediction_model' not in st.session_state:
        st.info("最初にモデルを構築してください")
        return
    
    # 評価指標の説明を追加
    if 'model_r2' in st.session_state and 'model_mse' in st.session_state:
        col1, col2 = st.columns(2)
        col1.metric("決定係数 (R²)", f"{st.session_state['model_r2']:.3f}")
        
        # MSEからRMSEを計算して表示
        rmse = np.sqrt(st.session_state['model_mse'])
        col2.metric("平均二乗誤差の平方根 (RMSE)", f"{rmse:,.0f}")
        
        with st.expander("評価指標の説明"):
            st.markdown("""
            ### 決定係数 (R²)
            決定係数は0から1の値をとり、1に近いほど予測精度が高いことを示します。例えば、0.75という値は、モデルが価格変動の75%を説明できることを意味します。
            
            ### 平均二乗誤差の平方根 (RMSE)
            RMSEは予測値と実際の値の差（誤差）の二乗平均の平方根です。この値が小さいほど予測精度が高いことを示します。
            
            RMSEは実際の予測誤差の目安となる値で、例えばRMSEが300万円の場合、予測値は平均して実際の価格から±300万円程度ずれる可能性があることを意味します。単位は「円」なので、直感的に理解しやすい指標です。
            """)
    
    st.subheader("価格予測")
    st.write("特徴量の値を入力して物件価格を予測します")
    
    # フォームを使用して予測入力と実行ボタンをグループ化
    with st.form(key='prediction_form'):
        # 特徴量の入力フォーム
        original_features = st.session_state.get('original_features', [])
        categorical_features = st.session_state.get('categorical_features', [])
        
        # 数値特徴量とカテゴリカル特徴量を分離
        numeric_features = [f for f in original_features if f not in categorical_features]
        
        input_values = {}
        
        # 数値特徴量の入力フォーム
        if numeric_features:
            st.subheader("数値特徴量")
            cols = st.columns(len(numeric_features))
            
            for i, feature in enumerate(numeric_features):
                if feature == 'Area':
                    input_values[feature] = cols[i].number_input(
                        "面積 (m²)", 
                        min_value=10.0, 
                        max_value=500.0, 
                        value=70.0 if 'input_values' not in st.session_state else st.session_state['input_values'].get(feature, 70.0), 
                        step=5.0,
                        key=f"pred_{feature}"
                    )
                elif feature == 'BuildingAge':
                    input_values[feature] = cols[i].number_input(
                        "築年数 (年)", 
                        min_value=0, 
                        max_value=50, 
                        value=15 if 'input_values' not in st.session_state else st.session_state['input_values'].get(feature, 15), 
                        step=1,
                        key=f"pred_{feature}"
                    )
        
        # カテゴリカル特徴量の入力フォーム
        if categorical_features:
            st.subheader("カテゴリカル特徴量")
            cols = st.columns(len(categorical_features))
            
            for i, feature in enumerate(categorical_features):
                if feature == 'Type':
                    # データから物件タイプの選択肢を取得
                    available_types = st.session_state.get('available_types', ['マンション', '戸建', '土地'])
                    input_values[feature] = cols[i].selectbox(
                        "物件タイプ",
                        options=available_types,
                        index=0 if 'input_values' not in st.session_state else available_types.index(st.session_state['input_values'].get(feature, available_types[0])) if st.session_state['input_values'].get(feature) in available_types else 0,
                        key=f"pred_{feature}"
                    )
                elif feature == 'Renovation':
                    # 改装情報の選択肢
                    available_renovations = st.session_state.get('available_renovations', ['未改装', '改装済'])
                    input_values[feature] = cols[i].selectbox(
                        "改装状況",
                        options=available_renovations,
                        index=0 if 'input_values' not in st.session_state else available_renovations.index(st.session_state['input_values'].get(feature, available_renovations[0])) if st.session_state['input_values'].get(feature) in available_renovations else 0,
                        key=f"pred_{feature}"
                    )
        
        # フォーム内の予測ボタン
        submitted = st.form_submit_button('予測実行')
        
        # 入力値をセッションに保存
        st.session_state['input_values'] = input_values
    
    # フォームが送信されたら予測実行
    if submitted:
        predicted_price = predict_with_model(st.session_state['input_values'])
        if predicted_price is not None:
            from utils import format_price
            st.success(f"予測価格: {format_price(predicted_price)}")
            
            # 信頼区間（簡易的なもの）
            lower_bound = predicted_price * 0.8
            upper_bound = predicted_price * 1.2
            st.write(f"予測範囲: {format_price(lower_bound)} 〜 {format_price(upper_bound)}")
            
            # 予測結果の解釈
            show_prediction_interpretation(predicted_price, df, st.session_state['original_features'])
    
    # 以前の予測結果がある場合も表示
    elif 'prediction_result' in st.session_state:
        predicted_price = st.session_state['prediction_result']
        from utils import format_price
        st.success(f"予測価格: {format_price(predicted_price)}")
        
        # 信頼区間（簡易的なもの）
        lower_bound = predicted_price * 0.8
        upper_bound = predicted_price * 1.2
        st.write(f"予測範囲: {format_price(lower_bound)} 〜 {format_price(upper_bound)}")
        
        # 予測結果の解釈
        show_prediction_interpretation(predicted_price, df, st.session_state.get('original_features', []))

def show_prediction_interpretation(predicted_price, df, features):
    """予測価格の解釈を表示"""
    st.write("### 予測価格の解釈")
    
    # 特徴量の影響度の説明
    explanation_parts = []
    
    if 'Area' in features:
        explanation_parts.append("面積が広いほど価格は高くなる")
    
    if 'BuildingAge' in features:
        explanation_parts.append("築年数が古いほど価格は低くなる")
    
    if 'Type' in features:
        explanation_parts.append("物件タイプによって価格が大きく変動する")
    
    if 'Renovation' in features:
        explanation_parts.append("改装済みの物件は価格が高くなる傾向がある")
    
    if explanation_parts:
        explanation = "一般的に、" + "、".join(explanation_parts) + "傾向があります。"
        st.write(explanation)
    
    # 平均価格との比較
    avg_price = df['TradePrice'].mean()
    from utils import format_price
    if predicted_price > avg_price * 1.2:
        st.write(f"予測価格は平均価格（{format_price(avg_price)}）より**高い**値となっています。")
    elif predicted_price < avg_price * 0.8:
        st.write(f"予測価格は平均価格（{format_price(avg_price)}）より**低い**値となっています。")
    else:
        st.write(f"予測価格は平均価格（{format_price(avg_price)}）と**同程度**の値となっています。")