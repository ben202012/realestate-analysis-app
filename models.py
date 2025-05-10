# models.py
"""
機械学習モデルを利用した不動産価格予測機能を提供するモジュール
scikit-learnモデルを使用したRMSE（平均二乗誤差の平方根）の改善を実現
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

def train_price_prediction_model(df, features, model_type='auto'):
    """
    価格予測モデルを構築する関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        学習データ
    features : list
        使用する特徴量のリスト
    model_type : str, optional
        使用するモデルの種類 ('rf', 'gbm', 'ensemble', 'auto')
        'auto'を指定すると自動的に最適なモデルを選択
        
    Returns:
    --------
    model : trained model object
        学習済みモデル
    mse : float
        テストデータの平均二乗誤差
    r2 : float
        テストデータの決定係数
    """
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
    
    # デバッグモードの確認
    debug_mode = st.session_state.get('debug_mode', False)
    
    # 元の特徴量を保存
    st.session_state['original_features'] = features
    
    # カテゴリカル変数の処理
    categorical_features = ['Type', 'Renovation', 'Structure', 'CityPlanning']
    categorical_features_present = [f for f in categorical_features if f in features and f in model_df.columns]
    
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

    # 地域特徴量の重み付け
    district_feature_importance = st.session_state.get('district_feature_importance', 5)
    if district_feature_importance > 1:
        district_columns = [col for col in X.columns if col.startswith('District') or col.startswith('Area_')]
        
        if district_columns and debug_mode:
            st.write(f"{len(district_columns)}個の地域特徴量に重み付けを適用します（係数: {district_feature_importance}）")
        
        # 特徴量の重み付け方法1: 該当列を複製して影響を高める
        if district_feature_importance <= 5:
            for _ in range(district_feature_importance - 1):
                for col in district_columns:
                    X[f"{col}_weighted_{_+1}"] = X[col]
        
        # 特徴量の重み付け方法2: サンプルの複製（より強い効果）
        else:
            # オリジナルデータを保持
            X_orig = X.copy()
            y_orig = y.copy()
            
            # 重要度に応じてサンプルを複製
            for _ in range(district_feature_importance - 5):
                X = pd.concat([X, X_orig], ignore_index=True)
                y = pd.concat([y, y_orig], ignore_index=True)
            
            if debug_mode:
                st.write(f"データセットのサイズが {len(X_orig)} から {len(X)} に増加しました")
    
    # メモリ使用量を最適化
    X = X.astype({col: 'float32' for col in X.select_dtypes(['float64']).columns})
    if debug_mode:
        st.write("メモリ使用量を最適化しました")
    
    # 地理的特徴量の追加（市区町村レベルの特徴量）
    if 'DistrictName' in model_df.columns and 'DistrictName' not in features:
        # 地区別の平均価格を計算
        district_avg = model_df.groupby('DistrictName')['TradePrice'].mean().to_dict()
        
        # 各データポイントに地区の平均価格を特徴量として追加
        X['DistrictAvgPrice'] = model_df['DistrictName'].map(district_avg)
        model_features.append('DistrictAvgPrice')
        
        if debug_mode:
            st.write("地区別の平均価格を特徴量として追加しました")
    
    # スケーリング
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_features) > 0:
        scaler = RobustScaler()
        X_scaled = X.copy()
        X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    else:
        X_scaled = X
    
    # トレーニングデータとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # プログレスバーの初期化
    progress_text = "モデルをトレーニング中..."
    progress_bar = st.progress(0)
    
    # 使用可能なモデルの定義
    available_models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42)
    }
    
    # モデルの選択とトレーニング
    if model_type == 'auto':
        # 複数のモデルを試して最適なものを選択
        st.info("複数のモデルを評価中です。これには少し時間がかかる場合があります...")
        
        # 各モデルの評価
        model_scores = {}
        
        for i, (name, model) in enumerate(available_models.items()):
            # k-fold cross validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_root_mean_squared_error')
                mean_score = -cv_scores.mean()  # RMSEなので負の値を正の値に変換
                model_scores[name] = mean_score
            except Exception as e:
                if debug_mode:
                    st.write(f"{name}の評価中にエラーが発生しました: {str(e)}")
                continue
            
            # プログレスバー更新
            progress_bar.progress((i + 1) / len(available_models))
        
        if not model_scores:
            st.error("すべてのモデル評価が失敗しました。RandomForestを使用します。")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            # 最適なモデルを選択
            best_model_name = min(model_scores, key=model_scores.get)
            if debug_mode:
                st.write(f"各モデルのRMSE: {model_scores}")
                st.write(f"最適なモデル: {best_model_name}, RMSE: {model_scores[best_model_name]:,.0f}")
            
            # 最適なモデルをデフォルトのパラメータで初期化
            model = available_models[best_model_name]
            
            # 最適なモデルでハイパーパラメータチューニングを行う
            if len(X_train) >= 100:  # データ量が十分ある場合のみ
                st.info(f"{best_model_name}モデルのハイパーパラメータをチューニング中...")
                
                # モデル別のハイパーパラメータグリッド
                if best_model_name == 'RandomForest':
                    param_grid = {
                        'n_estimators': [50, 100, 150],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                elif best_model_name == 'GradientBoosting':
                    param_grid = {
                        'n_estimators': [50, 100, 150],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                elif best_model_name == 'ExtraTrees':
                    param_grid = {
                        'n_estimators': [50, 100, 150],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                else:  # Ridge or Lasso
                    param_grid = {
                        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                    }
                
                # グリッドサーチの実行
                try:
                    grid_search = GridSearchCV(
                        model, param_grid, cv=3, 
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    
                    if debug_mode:
                        st.write(f"最適なパラメータ: {grid_search.best_params_}")
                        st.write(f"チューニング後のRMSE: {-grid_search.best_score_:,.0f}")
                except Exception as e:
                    if debug_mode:
                        st.write(f"ハイパーパラメータチューニング中にエラーが発生しました: {str(e)}")
                    st.warning("ハイパーパラメータチューニングをスキップします。デフォルトパラメータを使用します。")
    
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=150, 
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    elif model_type == 'ensemble':
        # アンサンブルモデル（複数のモデルの平均）
        models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(random_state=42),
            ExtraTreesRegressor(n_estimators=100, random_state=42),
        ]
        
        # 各モデルを個別にトレーニング
        trained_models = []
        for i, m in enumerate(models):
            try:
                m.fit(X_train, y_train)
                trained_models.append(m)
            except Exception as e:
                if debug_mode:
                    st.write(f"モデルトレーニング中にエラーが発生しました: {str(e)}")
                continue
                
            progress_bar.progress((i + 1) / (len(models) + 1))
        
        if not trained_models:
            st.error("すべてのモデルトレーニングが失敗しました。単一のRandomForestを使用します。")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['is_ensemble'] = False
        else:
            # アンサンブルモデルとして保存
            st.session_state['ensemble_models'] = trained_models
            st.session_state['is_ensemble'] = True
            
            # アンサンブル予測のためのダミーモデル
            model = RandomForestRegressor()
    else:
        # デフォルトはRandomForest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # モデルのトレーニング
    if model_type != 'ensemble' or not st.session_state.get('is_ensemble', False):
        st.session_state['is_ensemble'] = False
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"モデルトレーニング中にエラーが発生しました: {str(e)}")
            st.warning("代替のRandomForestモデルを使用します")
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
        progress_bar.progress(0.8)
    
    # テストデータでの予測
    if st.session_state.get('is_ensemble', False):
        # 各モデルの予測値の平均
        y_pred = np.mean([m.predict(X_test) for m in st.session_state['ensemble_models']], axis=0)
    else:
        y_pred = model.predict(X_test)
    
    # モデル評価
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # モデルと特徴量をセッション状態に保存
    st.session_state['prediction_model'] = model
    st.session_state['model_features'] = model_features
    st.session_state['categorical_features'] = categorical_features_present
    st.session_state['model_r2'] = r2
    st.session_state['model_mse'] = mse
    st.session_state['model_rmse'] = rmse
    st.session_state['model_mae'] = mae
    st.session_state['scaler'] = scaler if len(numeric_features) > 0 else None
    st.session_state['numeric_features'] = numeric_features
    
    # プログレスバー完了
    progress_bar.progress(1.0)
    time.sleep(0.5)  # 完了を表示するための短い待機
    progress_bar.empty()
    
    return model, mse, r2

def predict_price(model, input_features, feature_names):
    """
    ユーザー入力値から価格を予測する関数
    (app.pyとの互換性のために維持)
    
    Parameters:
    -----------
    model : trained model
        学習済みモデル
    input_features : dict
        入力特徴量の値
    feature_names : list
        特徴量の名前リスト
        
    Returns:
    --------
    float
        予測価格
    """
    if model is None:
        return None
    
    # 入力値をDataFrameに変換
    input_df = pd.DataFrame([input_features], columns=feature_names)
    
    # アンサンブルモデルの場合
    if st.session_state.get('is_ensemble', False) and 'ensemble_models' in st.session_state:
        # 各モデルの予測値の平均
        predictions = [m.predict(input_df)[0] for m in st.session_state['ensemble_models']]
        return np.mean(predictions)
    else:
        # 単一モデルの場合
        return model.predict(input_df)[0]

def predict_with_model(input_values=None):
    """
    セッションに保存されたモデルを使用して予測を実行する関数
    
    Parameters:
    -----------
    input_values : dict, optional
        予測に使用する特徴量の値
        
    Returns:
    --------
    float
        予測価格
    """
    # モデルと特徴量がセッションにあるか確認
    if 'prediction_model' not in st.session_state or 'model_features' not in st.session_state:
        st.error("先にモデルを構築してください")
        return None
    
    # アンサンブルモデルかどうかを確認
    is_ensemble = st.session_state.get('is_ensemble', False)
    
    if is_ensemble and 'ensemble_models' in st.session_state:
        models = st.session_state['ensemble_models']
        if not models:
            st.error("アンサンブルモデルが見つかりません")
            return None
    else:
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
    
    # 特定の特殊特徴量の処理
    # 地区平均価格を追加（モデルに含まれている場合）
    if 'DistrictAvgPrice' in features and 'DistrictAvgPrice' not in input_df.columns:
        input_df['DistrictAvgPrice'] = st.session_state.get('avg_district_price', 0)
    
    # モデルが使用する特徴量のみを選択
    try:
        input_df = input_df[features]
    except KeyError as e:
        st.error(f"特徴量の不整合: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.write("モデル特徴量:", features)
            st.write("入力特徴量:", input_df.columns.tolist())
        return None
    
    # 数値特徴量のスケーリング
    scaler = st.session_state.get('scaler')
    numeric_features = st.session_state.get('numeric_features', [])
    
    if scaler is not None and numeric_features:
        # 入力データフレームに存在する数値特徴量のみをスケーリング
        valid_numeric_features = [f for f in numeric_features if f in input_df.columns]
        if valid_numeric_features:
            input_df[valid_numeric_features] = scaler.transform(input_df[valid_numeric_features])
    
    # 予測実行
    try:
        if is_ensemble and 'ensemble_models' in st.session_state:
            # 各モデルの予測値の平均
            predictions = [m.predict(input_df)[0] for m in models]
            prediction = np.mean(predictions)
            
            # 予測値の分散も計算（信頼区間の推定に使用）
            prediction_std = np.std(predictions)
            st.session_state['prediction_std'] = prediction_std
        else:
            prediction = model.predict(input_df)[0]
        
        # 予測結果をセッションに保存
        st.session_state['prediction_result'] = prediction
        return prediction
    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {str(e)}")
        if st.session_state.get('debug_mode', False):
            import traceback
            st.write(traceback.format_exc())
        return None

def show_feature_importance(model, feature_names):
    """
    特徴量の重要度を表示する関数
    
    Parameters:
    -----------
    model : trained model
        学習済みモデル
    feature_names : list
        特徴量の名前リスト
        
    Returns:
    --------
    pandas.DataFrame
        特徴量重要度のDataFrame
    """
    if model is None:
        return None
    
    # アンサンブルモデルの場合は最初のモデル（RandomForest）の特徴量重要度を表示
    if st.session_state.get('is_ensemble', False):
        models = st.session_state.get('ensemble_models', [])
        tree_models = [m for m in models if hasattr(m, 'feature_importances_')]
        if not tree_models:
            st.info("このアンサンブルモデルには特徴量重要度を持つモデルが含まれていません")
            return None
        
        # 各モデルの特徴量重要度の平均を取る
        importances = np.mean([m.feature_importances_ for m in tree_models], axis=0)
    elif not hasattr(model, 'feature_importances_'):
        st.info("このモデルタイプでは特徴量重要度を表示できません")
        return None
    else:
        # 特徴量の重要度を取得
        importances = model.feature_importances_
    
    # 重要度をDataFrameに変換
    importance_df = pd.DataFrame({
        '特徴量': feature_names,
        '重要度': importances
    }).sort_values('重要度', ascending=False)
    
    # 上位10個の特徴量だけを表示
    if len(importance_df) > 10:
        plot_df = importance_df.head(10)
        title = '特徴量の重要度（上位10個）'
    else:
        plot_df = importance_df
        title = '特徴量の重要度'
    
    # 可視化
    fig = px.bar(
        plot_df,
        x='重要度',
        y='特徴量',
        title=title,
        orientation='h'  # 水平棒グラフ
    )
    
    # レイアウト調整
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},  # 重要度が低い順に並べる
        xaxis_title='重要度',
        yaxis_title=None,
        height=400,
        margin={'l': 10, 'r': 10, 't': 50, 'b': 10}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 特徴量の解釈表示
    with st.expander("特徴量の重要度について"):
        st.write("""
        ### 特徴量の重要度とは
        
        特徴量の重要度は、予測モデルが価格を予測する際に各特徴量（変数）がどの程度重要であるかを示します。
        
        - **値が大きいほど**、その特徴量が価格予測に**より重要**であることを示します。
        - 重要度は相対的なもので、すべての特徴量の重要度の合計は1になります。
        
        ### 不動産価格における一般的な重要特徴量
        
        一般的に不動産価格において重要な特徴量には以下のようなものがあります：
        
        - **立地（エリア/地区）**: 特定の地区は他よりも価値が高いか低い傾向があります
        - **面積**: 通常、広いほど価格は高くなります
        - **物件タイプ**: マンション、戸建て、土地などで価格傾向が異なります
        - **築年数**: 通常、新しい物件ほど価格が高くなる傾向があります
        
        モデルが学習した重要度は、対象地域の不動産市場の特性を反映しています。
        """)
    
    # 特徴量重要度をセッションに保存（予測解釈で使用）
    st.session_state['feature_importances_df'] = importance_df
    
    return importance_df

def show_prediction_interpretation(predicted_price, df, features):
    """
    予測価格の解釈を表示する関数
    
    Parameters:
    -----------
    predicted_price : float
        予測された価格
    df : pandas.DataFrame
        元データ
    features : list
        使用した特徴量のリスト
    """
    st.write("### 予測価格の解釈")
    
    # 特徴量の影響度に基づく説明
    explanation_parts = []
    
    # アンサンブルモデルの場合の追加説明
    if st.session_state.get('is_ensemble', False):
        st.write("この予測は複数のモデルの結果を組み合わせた**アンサンブル学習**による結果です。アンサンブル学習は単一のモデルよりも予測精度が高い傾向があります。")
    
    # セッションに保存された特徴量重要度を使って説明を構築
    # 特徴量ごとに影響度を説明
    if 'Area' in features:
        explanation_parts.append("**面積**が広いほど価格は高くなる")
    
    if 'BuildingAge' in features:
        explanation_parts.append("**築年数**が古いほど価格は低くなる")
    
    if 'Type' in features:
        explanation_parts.append("**物件タイプ**によって価格が大きく変動する")
    
    if 'Renovation' in features:
        explanation_parts.append("**改装済み**の物件は価格が高くなる傾向がある")
    
    if 'Structure' in features:
        explanation_parts.append("**構造**（木造、RC造など）によって耐久性や価値が異なる")
    
    if 'DistrictName' in features or 'DistrictAvgPrice' in features:
        explanation_parts.append("**地区・立地**が価格に大きく影響する")
    
    if 'DistrictPriceCluster' in features:
        explanation_parts.append("**地域のクラスタリング**による価格帯の類似性を考慮")
    
    if 'DistanceFromStation' in features:
        explanation_parts.append("**駅からの距離**が近いほど価格は高くなる")
    
    # 地域特徴語に関する説明
    area_features = [f for f in features if f.startswith('Area_')]
    if area_features:
        explanation_parts.append("**地名の特徴**が価格に影響する")
    
    if explanation_parts:
        explanation = "このモデルは以下の要素を考慮して予測を行っています：\n\n" + "\n".join([f"- {part}" for part in explanation_parts])
        st.write(explanation)
    
    # 平均価格との比較
    avg_price = df['TradePrice'].mean()
    median_price = df['TradePrice'].median()
    from utils import format_price
    
    # カラム分割で視覚的に比較しやすくする
    col1, col2, col3 = st.columns(3)
    col1.metric("予測価格", format_price(predicted_price))
    col2.metric("平均価格", format_price(avg_price), f"{(predicted_price - avg_price) / avg_price:.1%}")
    col3.metric("中央値", format_price(median_price), f"{(predicted_price - median_price) / median_price:.1%}")
    
    # 市場位置づけの解説
    st.write("### 市場位置づけ")
    
    # パーセンタイルの計算
    percentile = (df['TradePrice'] < predicted_price).mean() * 100
    
    # 予測価格の市場内での位置づけを表示
    if percentile > 90:
        market_position = "**非常に高額な**市場セグメントに位置しています"
    elif percentile > 75:
        market_position = "**高額な**市場セグメントに位置しています"
    elif percentile > 60:
        market_position = "**やや高め**の市場セグメントに位置しています"
    elif percentile > 40:
        market_position = "**平均的な**市場セグメントに位置しています"
    elif percentile > 25:
        market_position = "**やや安め**の市場セグメントに位置しています"
    elif percentile > 10:
        market_position = "**安価な**市場セグメントに位置しています"
    else:
        market_position = "**非常に安価な**市場セグメントに位置しています"
    
    st.write(f"この予測価格は市場全体の約**{percentile:.1f}パーセンタイル**に位置し、{market_position}。")
    
    # 物件タイプ別の比較（物件タイプが特徴量に含まれる場合）
    if 'Type' in features and 'input_values' in st.session_state and 'Type' in st.session_state['input_values']:
        property_type = st.session_state['input_values']['Type']
        type_data = df[df['Type'] == property_type]
        
        if not type_data.empty:
            type_avg = type_data['TradePrice'].mean()
            type_median = type_data['TradePrice'].median()
            type_percentile = (type_data['TradePrice'] < predicted_price).mean() * 100
            
            st.write(f"### {property_type}タイプとの比較")
            
            col1, col2 = st.columns(2)
            col1.metric(f"{property_type}の平均価格", format_price(type_avg), f"{(predicted_price - type_avg) / type_avg:.1%}")
            col2.metric(f"{property_type}の中央値", format_price(type_median), f"{(predicted_price - type_median) / type_median:.1%}")
            
            st.write(f"この予測価格は**{property_type}**カテゴリの中で約**{type_percentile:.1f}パーセンタイル**に位置しています。")
    
    # 築年数による価格変動の傾向を示す（築年数が特徴量に含まれる場合）
    if 'BuildingAge' in features and 'input_values' in st.session_state and 'BuildingAge' in st.session_state['input_values']:
        building_age = st.session_state['input_values']['BuildingAge']
        
        # 築年数と価格の関係を可視化
        with st.expander("築年数と価格の関係"):
            try:
                # 築年数と価格の関係を示す散布図
                age_fig = px.scatter(
                    df.sample(min(500, len(df))),  # データ量が多い場合はサンプリング
                    x='BuildingAge',
                    y='TradePrice',
                    opacity=0.6,
                    title="築年数と価格の関係"
                )
                
                # 予測対象の築年数を縦線で表示
                age_fig.add_vline(x=building_age, line_dash="dash", line_color="red")
                
                # レイアウト調整
                age_fig.update_layout(
                    xaxis_title="築年数（年）",
                    yaxis_title="取引価格（円）",
                    height=400
                )
                
                st.plotly_chart(age_fig, use_container_width=True)
            except Exception as e:
                if st.session_state.get('debug_mode', False):
                    st.write(f"グラフ生成エラー: {str(e)}")
            
            # 築年数に関する解説
            st.write(f"築{building_age}年の物件は、新築物件と比較して価値が下がる傾向がありますが、立地や状態によっては高い価値を保つ場合もあります。")
    
    # 地域特徴量の影響を解説
    district_explanation = ""
    if 'DistrictPriceCluster' in features and 'input_values' in st.session_state and 'DistrictPriceCluster' in st.session_state['input_values']:
        cluster = st.session_state['input_values'].get('DistrictPriceCluster')
        district_data = df[df['DistrictPriceCluster'] == cluster]
        
        if not district_data.empty:
            cluster_avg = district_data['TradePrice'].mean()
            overall_avg = df['TradePrice'].mean()
            
            if cluster_avg > overall_avg * 1.2:
                district_explanation = f"選択された地域は全体平均より**約{(cluster_avg/overall_avg-1)*100:.1f}%高価格**なエリアに属しています。"
            elif cluster_avg < overall_avg * 0.8:
                district_explanation = f"選択された地域は全体平均より**約{(1-cluster_avg/overall_avg)*100:.1f}%低価格**なエリアに属しています。"
            else:
                district_explanation = "選択された地域は価格帯が**平均的なエリア**に属しています。"
    
    # 地区名の特徴語の影響を解説
    area_name_explanation = ""
    area_features = [f for f in features if f.startswith('Area_')]
    if area_features and 'feature_importances_df' in st.session_state:
        importance_df = st.session_state['feature_importances_df']
        
        # 重要な地域特徴語を抽出
        important_areas = []
        for feature in area_features:
            if feature in importance_df['特徴量'].values:
                imp_row = importance_df[importance_df['特徴量'] == feature]
                if not imp_row.empty and imp_row['重要度'].values[0] > 0.01:  # 重要度1%以上
                    area_name = feature.replace('Area_', '')
                    important_areas.append(area_name)
        
        if important_areas:
            area_name_explanation = f"このエリアでは特に **{'、'.join(important_areas)}** などの地名が価格に影響を与えています。"
    
    # 総合的な地域解説を追加
    if district_explanation or area_name_explanation:
        st.write("### 地域特性")
        if district_explanation:
            st.write(district_explanation)
        if area_name_explanation:
            st.write(area_name_explanation)
    
    # モデルの精度に関する注意点
    st.info("注意: この予測は過去のデータに基づいたモデルを使用しており、実際の市場価格とは異なる場合があります。正確な価格評価には、不動産鑑定士などの専門家にご相談ください。")


def show_prediction_form_and_results(df):
    """
    予測フォームと結果を表示する関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        元データ
    """
    # モデルが構築済みか確認
    if 'prediction_model' not in st.session_state:
        st.info("最初にモデルを構築してください")
        return
    
    # 評価指標の説明を追加
    if 'model_r2' in st.session_state and 'model_mse' in st.session_state:
        col1, col2, col3 = st.columns(3)
        col1.metric("決定係数 (R²)", f"{st.session_state['model_r2']:.3f}")
        
        # RMSEを表示
        rmse = st.session_state.get('model_rmse', np.sqrt(st.session_state['model_mse']))
        col2.metric("RMSE (円)", f"{rmse:,.0f}")
        
        # MAEも表示
        mae = st.session_state.get('model_mae', 0)
        col3.metric("MAE (円)", f"{mae:,.0f}")
        
        # アンサンブルモデルかどうかを表示
        is_ensemble = st.session_state.get('is_ensemble', False)
        if is_ensemble:
            st.info("現在のモデル: アンサンブルモデル（複数のモデルの組み合わせ）")
        
        with st.expander("評価指標の説明"):
            st.markdown("""
            ### 決定係数 (R²)
            決定係数は0から1の値をとり、1に近いほど予測精度が高いことを示します。例えば、0.75という値は、モデルが価格変動の75%を説明できることを意味します。
            
            ### 平均二乗誤差の平方根 (RMSE)
            RMSEは予測値と実際の値の差（誤差）の二乗平均の平方根です。この値が小さいほど予測精度が高いことを示します。
            
            RMSEは実際の予測誤差の目安となる値で、例えばRMSEが300万円の場合、予測値は平均して実際の価格から±300万円程度ずれる可能性があることを意味します。単位は「円」なので、直感的に理解しやすい指標です。
            
            ### 平均絶対誤差 (MAE)
            MAEは予測値と実際の値の差の絶対値の平均です。RMSEと異なり、大きな誤差に対してペナルティが小さくなります。
            MAEも単位は「円」で、より直感的な予測誤差の指標になります。
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
            
            # 面積の入力
            if 'Area' in numeric_features:
                input_values['Area'] = st.number_input(
                    "面積 (m²)", 
                    min_value=10.0, 
                    max_value=500.0, 
                    value=70.0 if 'input_values' not in st.session_state else st.session_state['input_values'].get('Area', 70.0), 
                    step=5.0,
                    key=f"pred_Area"
                )
            
            # 築年数の入力
            if 'BuildingAge' in numeric_features:
                input_values['BuildingAge'] = st.number_input(
                    "築年数 (年)", 
                    min_value=0, 
                    max_value=50, 
                    value=15 if 'input_values' not in st.session_state else st.session_state['input_values'].get('BuildingAge', 15), 
                    step=1,
                    key=f"pred_BuildingAge"
                )
            
            # 駅からの距離（もしデータにあれば）
            if 'DistanceFromStation' in numeric_features:
                input_values['DistanceFromStation'] = st.number_input(
                    "駅からの距離 (分)", 
                    min_value=1, 
                    max_value=60, 
                    value=10 if 'input_values' not in st.session_state else st.session_state['input_values'].get('DistanceFromStation', 10), 
                    step=1,
                    key=f"pred_DistanceFromStation"
                )
        
        # カテゴリカル特徴量の入力フォーム
        if categorical_features:
            st.subheader("カテゴリカル特徴量")
            
            col_count = min(len(categorical_features), 3)  # 最大3列
            cols = st.columns(col_count)
            
            for i, feature in enumerate(categorical_features):
                col_idx = i % col_count
                
                if feature == 'Type':
                    # データから物件タイプの選択肢を取得
                    available_types = st.session_state.get('available_types', ['マンション', '戸建', '土地'])
                    input_values[feature] = cols[col_idx].selectbox(
                        "物件タイプ",
                        options=available_types,
                        index=0 if 'input_values' not in st.session_state else available_types.index(st.session_state['input_values'].get(feature, available_types[0])) if st.session_state['input_values'].get(feature) in available_types else 0,
                        key=f"pred_{feature}"
                    )
                
                elif feature == 'Renovation':
                    # 改装情報の選択肢
                    available_renovations = st.session_state.get('available_renovations', ['未改装', '改装済'])
                    input_values[feature] = cols[col_idx].selectbox(
                        "改装状況",
                        options=available_renovations,
                        index=0 if 'input_values' not in st.session_state else available_renovations.index(st.session_state['input_values'].get(feature, available_renovations[0])) if st.session_state['input_values'].get(feature) in available_renovations else 0,
                        key=f"pred_{feature}"
                    )
                
                elif feature == 'Structure':
                    # 構造の選択肢
                    available_structures = st.session_state.get('available_structures', ['木造', 'RC', 'SRC', '鉄骨造'])
                    input_values[feature] = cols[col_idx].selectbox(
                        "構造",
                        options=available_structures,
                        index=0 if 'input_values' not in st.session_state else available_structures.index(st.session_state['input_values'].get(feature, available_structures[0])) if st.session_state['input_values'].get(feature) in available_structures else 0,
                        key=f"pred_{feature}"
                    )
                
                elif feature == 'CityPlanning':
                    # 都市計画区分の選択肢
                    available_cityplannings = st.session_state.get('available_cityplannings', ['第1種低層住居専用地域', '第1種中高層住居専用地域', '第2種住居地域', '商業地域', '準工業地域'])
                    input_values[feature] = cols[col_idx].selectbox(
                        "都市計画区分",
                        options=available_cityplannings,
                        index=0 if 'input_values' not in st.session_state else available_cityplannings.index(st.session_state['input_values'].get(feature, available_cityplannings[0])) if st.session_state['input_values'].get(feature) in available_cityplannings else 0,
                        key=f"pred_{feature}"
                    )
        
        # 地区情報（オプション）
        if 'DistrictName' in df.columns and 'DistrictAvgPrice' in st.session_state.get('model_features', []):
            districts = df['DistrictName'].dropna().unique().tolist()
            if districts:
                selected_district = st.selectbox(
                    "地区",
                    options=districts,
                    index=0,
                    key="pred_district"
                )
                
                # 選択された地区の平均価格を取得（モデルの特徴量に使用）
                district_data = df[df['DistrictName'] == selected_district]
                if not district_data.empty:
                    avg_price = district_data['TradePrice'].mean()
                    st.session_state['avg_district_price'] = avg_price
        
        # 入力値をセッションに保存
        st.session_state['input_values'] = input_values
        
        # 高度な予測設定（エキスパートユーザー向け）
        with st.expander("高度な予測設定"):
            # 信頼区間の幅を調整
            confidence_interval = st.slider(
                "信頼区間の幅", 
                min_value=60, 
                max_value=99, 
                value=90, 
                step=1,
                help="予測価格の周りに表示する信頼区間の幅です（例：90%の信頼区間）"
            )
            st.session_state['confidence_interval'] = confidence_interval
        
        # フォーム内の予測ボタン
        submitted = st.form_submit_button('予測実行')
    
    # フォームが送信されたら予測実行
    if submitted:
        predicted_price = predict_with_model(st.session_state['input_values'])
        if predicted_price is not None:
            from utils import format_price
            
            # 予測結果を表示
            st.success(f"予測価格: {format_price(predicted_price)}")
            
            # モデルの精度に基づいた信頼区間の計算
            rmse = st.session_state.get('model_rmse', 0)
            
            # 信頼区間の設定（デフォルトは90%）
            confidence_level = st.session_state.get('confidence_interval', 90) / 100
            
            # 標準正規分布の両側信頼区間の値を計算
            # 例：90%信頼区間なら約1.645、95%なら約1.96
            try:
                from scipy import stats
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
            except:
                # scipyが利用できない場合は、一般的な値を使用
                if confidence_level >= 0.99:
                    z_score = 2.576  # 99%信頼区間
                elif confidence_level >= 0.95:
                    z_score = 1.96   # 95%信頼区間
                elif confidence_level >= 0.90:
                    z_score = 1.645  # 90%信頼区間
                elif confidence_level >= 0.80:
                    z_score = 1.28   # 80%信頼区間
                else:
                    z_score = 1.0    # デフォルト値
            
            # アンサンブルモデルの場合は予測の標準偏差も考慮
            if st.session_state.get('is_ensemble', False) and 'prediction_std' in st.session_state:
                # アンサンブルモデルの予測分散を使用
                prediction_std = st.session_state['prediction_std']
                # RMSEとアンサンブル予測の分散を組み合わせた信頼区間
                margin_of_error = z_score * np.sqrt(rmse**2 + prediction_std**2)
            else:
                # 単一モデルの場合はRMSEのみを使用
                margin_of_error = z_score * rmse
            
            lower_bound = max(0, predicted_price - margin_of_error)
            upper_bound = predicted_price + margin_of_error
            
            # 信頼区間を表示
            st.write(f"{confidence_level*100:.0f}%信頼区間: {format_price(lower_bound)} 〜 {format_price(upper_bound)}")
            
            # 信頼区間の視覚化
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode = "number+gauge",
                value = predicted_price,
                number = {"prefix": "", "suffix": "円", "valueformat": ",.0f"},
                gauge = {
                    'axis': {'range': [lower_bound*0.8, upper_bound*1.2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [lower_bound*0.8, lower_bound], 'color': "lightgray"},
                        {'range': [lower_bound, upper_bound], 'color': "royalblue"},
                        {'range': [upper_bound, upper_bound*1.2], 'color': "lightgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': predicted_price
                    }
                },
                title = {'text': "予測価格と信頼区間"}
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=30, r=30, t=50, b=30),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 予測結果の解釈
            show_prediction_interpretation(predicted_price, df, st.session_state['original_features'])
    
        # 以前の予測結果がある場合も表示
        elif 'prediction_result' in st.session_state:
            predicted_price = st.session_state['prediction_result']
            from utils import format_price
            
            # 予測結果を表示
            st.success(f"予測価格: {format_price(predicted_price)}")
            
            # モデルの精度に基づいた信頼区間の計算
            rmse = st.session_state.get('model_rmse', 0)
            
            # 信頼区間の設定（デフォルトは90%）
            confidence_level = st.session_state.get('confidence_interval', 90) / 100
            
            # 標準正規分布の両側信頼区間の値を計算
            try:
                from scipy import stats
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
            except:
                # scipyが利用できない場合は、一般的な値を使用
                if confidence_level >= 0.99:
                    z_score = 2.576  # 99%信頼区間
                elif confidence_level >= 0.95:
                    z_score = 1.96   # 95%信頼区間
                elif confidence_level >= 0.90:
                    z_score = 1.645  # 90%信頼区間
                elif confidence_level >= 0.80:
                    z_score = 1.28   # 80%信頼区間
                else:
                    z_score = 1.0    # デフォルト値
            
            # アンサンブルモデルの場合は予測の標準偏差も考慮
            if st.session_state.get('is_ensemble', False) and 'prediction_std' in st.session_state:
                prediction_std = st.session_state['prediction_std']
                margin_of_error = z_score * np.sqrt(rmse**2 + prediction_std**2)
            else:
                margin_of_error = z_score * rmse
            
            lower_bound = max(0, predicted_price - margin_of_error)
            upper_bound = predicted_price + margin_of_error
            
            # 信頼区間を表示
            st.write(f"{confidence_level*100:.0f}%信頼区間: {format_price(lower_bound)} 〜 {format_price(upper_bound)}")
            
            # 予測結果の解釈
            show_prediction_interpretation(predicted_price, df, st.session_state.get('original_features', []))



