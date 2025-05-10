# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import japanize_matplotlib

# 自作モジュールのインポート
from data_handler import fetch_and_process_data, get_municipalities
from visualizers import (
    show_basic_stats, plot_price_histogram, plot_price_by_type,
    plot_district_prices, plot_price_vs_building_age, plot_price_pie_chart,
    plot_heatmap, plot_time_trend, plot_price_per_sqm
)
from models import train_price_prediction_model, show_feature_importance, show_prediction_form_and_results
from utils import get_prefecture_dict, get_city_options, format_price, filter_dataframe, generate_summary_report
from price_trends import show_quarterly_price_trends

# タブ選択の状態を保持するための関数
# app.pyファイル内のtrain_model_on_click関数の更新
def train_model_on_click(df, features, model_type='auto'):
    """
    モデル構築ボタンクリック時に実行される関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        学習データ
    features : list
        使用する特徴量のリスト
    model_type : str
        使用するモデルの種類
    """
    with st.spinner('モデルを構築中...'):
        # 高度なオプションがある場合は取得
        advanced_options = st.session_state.get('model_advanced_options', {})
        
        # デバッグモードの確認
        debug_mode = st.session_state.get('debug_mode', False)
        
        if debug_mode and advanced_options:
            st.write("高度なオプション:", advanced_options)
        
        # 学習データのサンプル数を表示
        if debug_mode:
            st.write(f"学習データのサンプル数: {len(df)}")
            st.write(f"使用する特徴量: {features}")
            st.write(f"モデルタイプ: {model_type}")
        
        # データ数が少ない場合は警告
        if len(df) < 50:
            st.warning(f"学習データが少ないため（{len(df)}件）、モデルの精度が低くなる可能性があります。可能であればより多くのデータを取得してください。")
        
        # モデルの訓練
        model, mse, r2 = train_price_prediction_model(df, features, model_type)
        
        # 訓練結果の保存
        if model is not None:
            st.session_state['model_r2'] = r2
            st.session_state['model_mse'] = mse
            st.session_state['model_type'] = model_type
            
            # 特徴量の重要度を表示
            if hasattr(model, 'feature_importances_') or st.session_state.get('is_ensemble', False):
                st.subheader("特徴量の重要度")
                importance_df = show_feature_importance(model, st.session_state['model_features'])
                
                # 重要度の低い特徴量があれば警告
                if importance_df is not None and len(importance_df) > 3:
                    # 重要度が低い特徴量（全体の5%未満）を検出
                    low_importance = importance_df[importance_df['重要度'] < 0.05]
                    if not low_importance.empty:
                        st.info(f"以下の特徴量は重要度が低いため（5%未満）、次回のモデル構築時に除外を検討してください: {', '.join(low_importance['特徴量'].tolist())}")
        else:
            st.error("モデルの構築に失敗しました。データを確認してください。")
            
            # エラーの可能性がある場合の追加情報
            if len(features) == 0:
                st.error("特徴量が選択されていません。少なくとも1つ以上の特徴量を選択してください。")
            elif len(df) < 20:
                st.error("データ数が不足しています。モデル構築には少なくとも20件以上のデータが必要です。")
            
            # 解決策の提案
            st.info("解決策: 別の地域を選択する、より多くのデータを取得する、または異常値や欠損値の多い特徴量を除外してください。")

# ページ設定
st.set_page_config(
    page_title="不動産取引価格分析アプリ",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 0  # デフォルトタブのインデックス

# 検索条件の以前の値を保存するための状態変数
if 'previous_year' not in st.session_state:
    st.session_state['previous_year'] = None
if 'previous_prefecture' not in st.session_state:
    st.session_state['previous_prefecture'] = None
if 'previous_city' not in st.session_state:
    st.session_state['previous_city'] = None

# アプリのタイトルと説明
st.title('不動産取引価格分析アプリ')
st.write('国土交通省APIから取得した不動産取引データを分析・可視化するアプリケーションです')

# サイドバー - 検索条件
st.sidebar.header('検索条件')

# 年度選択
year = st.sidebar.selectbox(
    '年度',
    options=['2024', '2023', '2022', '2021', '2020', '2019'],
    index=0,  # 2024年をデフォルト選択
    key='year_select'  # キーを追加
)

# 都道府県選択
prefecture_dict = get_prefecture_dict()
prefecture_code = st.sidebar.selectbox(
    '都道府県',
    options=list(prefecture_dict.keys()),
    format_func=lambda x: prefecture_dict[x],
    index=list(prefecture_dict.keys()).index(27),  # 大阪府をデフォルト選択
    key='prefecture_select'  # キーを追加
)

# APIキー入力
api_key = st.sidebar.text_input('APIキー', type='password')

# 市区町村選択
if api_key:
    # 都道府県が選択されたら市区町村リストを取得
    with st.sidebar:
        with st.spinner('市区町村リストを取得中...'):
            municipalities = get_municipalities(api_key, prefecture_code)
            
            if municipalities:
                # 市区町村選択肢の作成
                city_options = [(int(m['id']), m['name']) for m in municipalities]  # 文字列を整数に変換
                city_code = st.selectbox(
                    '市区町村',
                    options=[opt[0] for opt in city_options],
                    format_func=lambda x: dict(city_options)[x],
                    index=0,
                    key='city_select'  # キーを追加
                )
            else:
                # API取得に失敗した場合、旧方式で表示
                city_options = get_city_options(prefecture_code)
                city_code = st.selectbox(
                    '市区町村',
                    options=[opt[0] for opt in city_options],
                    format_func=lambda x: dict(city_options)[x],
                    index=0,
                    key='city_select'  # キーを追加
                )
                st.warning("APIから市区町村リストを取得できなかったため、一部の市区町村のみ表示しています。")
else:
    # APIキーが未入力の場合は従来の方法でリスト表示
    city_options = get_city_options(prefecture_code)
    city_code = st.sidebar.selectbox(
        '市区町村',
        options=[opt[0] for opt in city_options],
        format_func=lambda x: dict(city_options)[x],
        index=0,
        key='city_select'  # キーを追加
    )

# 検索条件変更時にデータをリセットする処理
# 市区町村選択の変更検出を修正
conditions_changed = False

# 年度が変更されたか確認
if st.session_state['previous_year'] is not None and st.session_state['previous_year'] != year:
    conditions_changed = True
    if st.session_state.get('debug_mode', False):
        st.sidebar.info(f"年度が変更されました: {st.session_state['previous_year']} → {year}")
st.session_state['previous_year'] = year

# 都道府県が変更されたか確認
if st.session_state['previous_prefecture'] is not None and st.session_state['previous_prefecture'] != prefecture_code:
    conditions_changed = True
    if st.session_state.get('debug_mode', False):
        st.sidebar.info(f"都道府県が変更されました: {st.session_state['previous_prefecture']} → {prefecture_code}")
st.session_state['previous_prefecture'] = prefecture_code

# 市区町村が変更されたか確認
# ここでセッション変数を使用して比較
if 'city_select' in st.session_state:
    current_city = st.session_state['city_select']
    if st.session_state['previous_city'] is not None and st.session_state['previous_city'] != current_city:
        conditions_changed = True
        if st.session_state.get('debug_mode', False):
            st.sidebar.info(f"市区町村が変更されました: {st.session_state['previous_city']} → {current_city}")
    st.session_state['previous_city'] = current_city

# 条件が変更されていて、データがセッションに存在する場合、データをリセット
if conditions_changed and 'data' in st.session_state:
    del st.session_state['data']
    # モデル関連のセッションデータもリセット
    for key in ['prediction_model', 'model_features', 'model_r2', 'model_mse', 'model_rmse', 'prediction_result']:
        if key in st.session_state:
            del st.session_state[key]
    # リセットメッセージを表示
    st.warning('検索条件が変更されたため、データがリセットされました。新しい条件でデータを取得してください。')

# データ取得ボタン
if st.sidebar.button('データ取得'):
    if api_key:
        with st.spinner('データを取得中...'):
            # データ取得前に既存のデータをセッションから削除
            if 'data' in st.session_state:
                del st.session_state['data']
            
            df = fetch_and_process_data(api_key, year, prefecture_code, city_code)
            
            if df is not None and not df.empty:
                st.session_state['data'] = df
                st.success(f'{len(df)}件のデータを取得しました')
            else:
                # エラー時にはセッションからデータを削除することを再確認
                if 'data' in st.session_state:
                    del st.session_state['data']
                st.error('データを取得できませんでした')
    else:
        st.sidebar.error('APIキーを入力してください')

# サイドバー - フィルタリング
st.sidebar.header('データフィルタリング')
if 'data' in st.session_state:
    df = st.session_state['data']
    
    # フィルタリング条件の設定
    filters = {}
    
    # 物件タイプフィルタ
    if 'Type' in df.columns:
        types = ['すべて'] + list(df['Type'].dropna().unique())
        filters['property_type'] = st.sidebar.selectbox('物件タイプ', types)
    
    # 価格範囲フィルタ
    if 'TradePrice' in df.columns and not df['TradePrice'].isna().all():
        price_min = float(df['TradePrice'].min()) if not pd.isna(df['TradePrice'].min()) else 0.0
        price_max = float(df['TradePrice'].max()) if not pd.isna(df['TradePrice'].max()) else 1000000.0
        # 最小値と最大値が同じ場合、最大値を少し増やす
        if price_min == price_max:
            price_max = price_min + 1.0
        filters['price_range'] = st.sidebar.slider(
            '価格範囲（円）',
            min_value=int(price_min),
            max_value=int(price_max),
            value=(int(price_min), int(price_max)),
            step=100000  # 10万円単位でスライド
        )
    
    # 築年数範囲フィルタ
    if 'BuildingAge' in df.columns and not df['BuildingAge'].isna().all():
        age_min = float(df['BuildingAge'].min()) if not pd.isna(df['BuildingAge'].min()) else 0.0
        age_max = float(df['BuildingAge'].max()) if not pd.isna(df['BuildingAge'].max()) else 50.0
        # 最小値と最大値が同じ場合、最大値を少し増やす
        if age_min == age_max:
            age_max = age_min + 1.0
        filters['building_age_range'] = st.sidebar.slider(
            '築年数範囲（年）',
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max)
        )
    # サイドバーに外れ値処理設定を追加
    st.sidebar.header('データ処理設定')
    with st.sidebar.expander("外れ値処理の設定", expanded=False):
        # 外れ値除去を使用するかどうか
        use_outlier_removal = st.checkbox(
            "外れ値処理を有効にする", 
            value=True,
            help="外れ値（異常に高い/低い価格など）を除外します"
        )
        st.session_state['use_outlier_removal'] = use_outlier_removal
        
        if use_outlier_removal:
            # 外れ値検出方法
            outlier_method = st.selectbox(
                "検出方法",
                options=["iqr", "zscore", "dbscan"],
                format_func=lambda x: {
                    "iqr": "IQR法（四分位範囲法）",
                    "zscore": "Z-score法（標準偏差）",
                    "dbscan": "DBSCAN法（クラスタリング）"
                }.get(x, x),
                help="外れ値を検出するアルゴリズムを選択します"
            )
            st.session_state['outlier_method'] = outlier_method
            
            # 閾値の設定
            if outlier_method == "iqr":
                threshold = st.slider(
                    "IQR倍数", 
                    min_value=1.0, 
                    max_value=3.0, 
                    value=1.5, 
                    step=0.1,
                    help="大きいほど外れ値として判断される範囲が広くなります"
                )
            elif outlier_method == "zscore":
                threshold = st.slider(
                    "Z-score閾値", 
                    min_value=2.0, 
                    max_value=5.0, 
                    value=3.0, 
                    step=0.1,
                    help="大きいほど外れ値として判断される範囲が狭くなります"
                )
            else:  # dbscan
                threshold = st.slider(
                    "DBSCAN閾値", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.5, 
                    step=0.1,
                    help="クラスタリングの距離パラメータです"
                )
            
            st.session_state['outlier_threshold'] = threshold



    # フィルタリング適用
    filtered_df = filter_dataframe(df, filters)
    
    st.sidebar.write(f'フィルタリング結果: {len(filtered_df)}件 / 全{len(df)}件')
else:
    filtered_df = None

# サイドバー - 開発者モード (最後に移動)
st.sidebar.header('開発者オプション')
# セッション状態にデバッグモードが存在しない場合は初期化
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False
# チェックボックスの値をセッション状態に反映
debug_mode = st.sidebar.checkbox('デバッグモードを有効にする', value=st.session_state['debug_mode'])
st.session_state['debug_mode'] = debug_mode

# サイドバー - ヘルプと使い方
with st.sidebar.expander("ヘルプと使い方"):
    st.markdown("""
    ### 使い方ガイド
    
    1. **データ取得**
       - 年度、都道府県、市区町村を選択
       - APIキーを入力
       - 「データ取得」ボタンをクリック
    
    2. **データフィルタリング**
       - 物件タイプ、価格範囲、築年数範囲でデータを絞り込み
    
    3. **各タブの機能**
       - **基本統計**: 基本的な統計情報とデータテーブル
       - **価格分析**: 価格分布、物件タイプ別価格、築年数と価格の関係
       - **エリア分析**: 地区別の平均価格と件数
       - **高度な分析**: 相関分析、時系列分析など
       - **価格予測**: 機械学習モデルによる価格予測
       - **四半期推移**: 四半期ごとの価格推移分析
    
    4. **データのダウンロードと要約レポート**
       - CSVまたはExcel形式でデータをダウンロード
       - 要約レポートを生成してMarkdown形式でダウンロード
    
    ### APIキーについて
    
    国土交通省の不動産取引価格情報APIを使用するためには、APIキーが必要です。
    [国土交通省 不動産取引価格情報提供サイト](https://www.land.mlit.go.jp/webland/api.html)から取得できます。
    
    ### お問い合わせ
    
    ご質問やフィードバックは[こちら](mailto:your.email@example.com)までお願いします。
    """)

# メインコンテンツ
if 'data' in st.session_state and filtered_df is not None:
    # タブの設定
    tab_names = ['基本統計', '価格分析', 'エリア分析', '高度な分析', '価格予測', '四半期推移']
    selected_tab = st.radio("分析タブ", tab_names, index=st.session_state['current_tab'], horizontal=True, label_visibility="collapsed")
    st.session_state['current_tab'] = tab_names.index(selected_tab)

    # 選択したタブに応じてコンテンツを表示
    if selected_tab == '基本統計':
        st.header('基本統計情報')
        
        # 基本統計情報表示
        show_basic_stats(filtered_df)
        
        # データテーブル表示
        with st.expander('データテーブル表示'):
            st.dataframe(filtered_df)

    elif selected_tab == '価格分析':
        st.header('価格分析')
        
        # 価格分布ヒストグラム
        plot_price_histogram(filtered_df)
        
        # 価格帯別の割合（円グラフ）
        plot_price_pie_chart(filtered_df)
        
        # タイプ別の価格分布
        plot_price_by_type(filtered_df)
        
        # 築年数と価格の関係
        plot_price_vs_building_age(filtered_df)

    elif selected_tab == 'エリア分析':
        st.header('エリア分析')
        
        # 地区別の平均価格
        plot_district_prices(filtered_df)
        
        # 地区別のデータ件数
        if 'DistrictName' in filtered_df.columns:
            district_counts = filtered_df['DistrictName'].value_counts().reset_index()
            district_counts.columns = ['地区名', '件数']
            
            fig = px.bar(
                district_counts.head(10),
                x='地区名',
                y='件数',
                title='地区別の物件数（上位10地区）'
            )
            st.plotly_chart(fig, use_container_width=True)

    elif selected_tab == '高度な分析':
        st.header('高度な分析')
        
        # 相関分析
        st.subheader('相関分析')
        plot_heatmap(filtered_df)
        
        # 時系列トレンド
        st.subheader('時系列トレンド')
        plot_time_trend(filtered_df)
        
        # 平米単価の分析
        st.subheader('平米単価の分析')
        plot_price_per_sqm(filtered_df)
        
        # 構造別の分析
        if 'Structure' in filtered_df.columns and 'TradePrice' in filtered_df.columns:
            st.subheader('構造別の価格分析')
            structure_price = filtered_df.groupby('Structure')['TradePrice'].agg(['mean', 'count']).reset_index()
            structure_price = structure_price.sort_values('mean', ascending=False)
            
            fig = px.bar(
                structure_price,
                x='Structure',
                y='mean',
                text='count',
                title='構造別の平均価格',
                labels={'Structure': '構造', 'mean': '平均価格（円）', 'count': '件数'}
            )
            st.plotly_chart(fig, use_container_width=True)

    elif selected_tab == '価格予測':
        st.header('価格予測モデル')
        st.write('人工知能（機械学習）を使用した価格予測モデルを構築します。')
        
        # 特徴量選択
        available_features = ['Area', 'BuildingAge', 'Type', 'Renovation', 'Structure', 'DistrictName']
        # データに存在する特徴量のみを表示
        available_features = [f for f in available_features if f in filtered_df.columns]
        
        selected_features = st.multiselect(
            '使用する特徴量を選択',
            options=available_features,
            default=[f for f in ['Area', 'BuildingAge', 'Type'] if f in available_features]
        )
        
        # モデルタイプの選択
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "モデルタイプ",
                options=["auto", "rf", "gbm", "xgb", "lgb", "ensemble"],
                format_func=lambda x: {
                    "auto": "自動選択（推奨）", 
                    "rf": "ランダムフォレスト", 
                    "gbm": "勾配ブースティング", 
                    "xgb": "XGBoost", 
                    "lgb": "LightGBM", 
                    "ensemble": "アンサンブル学習"
                }.get(x, x),
                index=0,
                help="自動選択では複数のモデルをテストして最適なものを選びます。アンサンブル学習は複数のモデルを組み合わせて精度を向上させます。"
            )
        
        with col2:
            advanced_options = st.checkbox("高度なオプションを表示", value=False)
        
        # 高度なオプション
        if advanced_options:
            with st.expander("高度なモデル設定"):
                # 交差検証のfold数
                cv_folds = st.slider(
                    "交差検証の分割数", 
                    min_value=3, 
                    max_value=10, 
                    value=5, 
                    help="交差検証はモデルの汎化性能を評価するためのテクニックです。大きい数を選ぶとより信頼性の高い評価ができますが、計算時間が長くなります。"
                )
                
                # テストデータの割合
                test_size = st.slider(
                    "テストデータの割合", 
                    min_value=0.1, 
                    max_value=0.5, 
                    value=0.2, 
                    step=0.05,
                    help="モデル評価に使用するデータの割合です。通常は0.2（20%）程度が適切です。"
                )
                
                # ハイパーパラメータチューニング
                hyperparameter_tuning = st.checkbox(
                    "ハイパーパラメータチューニングを実行", 
                    value=True,
                    help="モデルの最適なパラメータを自動的に探索します。計算時間が長くなる場合があります。"
                )
                
                st.session_state['model_advanced_options'] = {
                    'cv_folds': cv_folds,
                    'test_size': test_size,
                    'hyperparameter_tuning': hyperparameter_tuning
                }
        
        # フォームを使用してモデル構築ボタンをグループ化
        with st.form(key='model_build_form'):
            submit_build = st.form_submit_button('モデル構築')
            
        if submit_build and len(selected_features) > 0:
            with st.spinner(f'選択したモデル（{model_type}）を構築中... これには数分かかる場合があります'):
                # モデル構築関数を呼び出し
                train_model_on_click(filtered_df, selected_features, model_type=model_type)
                
                # 評価指標を表示
                if 'model_r2' in st.session_state:
                    col1, col2, col3 = st.columns(3)
                    
                    # R²スコア
                    r2 = st.session_state['model_r2']
                    r2_color = 'normal'
                    if r2 > 0.8:
                        r2_color = 'good'
                    elif r2 < 0.5:
                        r2_color = 'off'
                    col1.metric("決定係数 (R²)", f"{r2:.3f}", delta_color=r2_color)
                    
                    # RMSE
                    rmse = st.session_state.get('model_rmse', np.sqrt(st.session_state['model_mse']))
                    # 平均価格に対するRMSEの割合
                    avg_price = filtered_df['TradePrice'].mean()
                    rmse_ratio = rmse / avg_price
                    rmse_status = f"平均価格の{rmse_ratio:.1%}"
                    rmse_color = 'normal'
                    if rmse_ratio < 0.15:
                        rmse_color = 'good'
                    elif rmse_ratio > 0.3:
                        rmse_color = 'off'
                    col2.metric("RMSE", f"{rmse:,.0f}", rmse_status, delta_color=rmse_color)
                    
                    # MAE
                    mae = st.session_state.get('model_mae', 0)
                    col3.metric("MAE", f"{mae:,.0f}")
                    
                    # モデルタイプの表示
                    if st.session_state.get('is_ensemble', False):
                        st.success("アンサンブルモデル（複数のアルゴリズムの組み合わせ）を使用しています。より精度の高い予測が可能です。")
                    
                    # モデル品質の評価
                    model_quality = ""
                    if r2 > 0.85:
                        model_quality = "非常に高い（R² > 0.85）"
                        st.success(f"モデルの品質: {model_quality}")
                    elif r2 > 0.75:
                        model_quality = "良好（R² > 0.75）"
                        st.success(f"モデルの品質: {model_quality}")
                    elif r2 > 0.6:
                        model_quality = "許容範囲内（R² > 0.6）"
                        st.info(f"モデルの品質: {model_quality}")
                    elif r2 > 0.5:
                        model_quality = "限定的（R² > 0.5）"
                        st.warning(f"モデルの品質: {model_quality} - 特徴量の追加をご検討ください")
                    else:
                        model_quality = "不十分（R² < 0.5）"
                        st.error(f"モデルの品質: {model_quality} - より多くの特徴量やデータが必要です")
        
        # モデル構築後の予測フォーム表示
        if 'prediction_model' in st.session_state:
            # 予測フォームと結果を表示
            show_prediction_form_and_results(filtered_df)    
        

            # 地域特徴量の設定
            with st.sidebar.expander("地域特徴量の設定", expanded=False):
                use_district_features = st.checkbox(
                    "地域特徴量を使用する", 
                    value=True,
                    help="地名による特徴抽出と地域クラスタリングを行います"
                )
                st.session_state['use_district_features'] = use_district_features
                
                if use_district_features:
                    # 地域特徴量の詳細設定
                    district_feature_importance = st.slider(
                        "地域特徴量の重要度", 
                        min_value=1, 
                        max_value=10, 
                        value=5,
                        help="モデルにおける地域特徴量の重みづけを調整します（値が大きいほど地域の影響が強くなります）"
                    )
                    st.session_state['district_feature_importance'] = district_feature_importance




    elif selected_tab == '四半期推移':
        # 四半期推移タブの内容
        show_quarterly_price_trends(api_key, prefecture_code, city_code, year=year)
    # データのダウンロード機能
    st.header('データのダウンロード')
    
    # CSVダウンロードボタン
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="CSVでダウンロード",
        data=csv,
        file_name="real_estate_data.csv",
        mime="text/csv",
    )
    
    # Excelダウンロードボタン
    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, sheet_name='不動産データ', index=False)
        # シートを追加して統計情報を書き込む
        stats_df = pd.DataFrame({
            '統計量': ['件数', '平均価格', '中央値', '最小値', '最大値', '標準偏差'],
            '値': [
                len(filtered_df),
                filtered_df['TradePrice'].mean(),
                filtered_df['TradePrice'].median(),
                filtered_df['TradePrice'].min(),
                filtered_df['TradePrice'].max(),
                filtered_df['TradePrice'].std()
            ]
        })
        stats_df.to_excel(writer, sheet_name='統計情報', index=False)
    
    excel_data = buffer.getvalue()
    st.download_button(
        label="Excelでダウンロード",
        data=excel_data,
        file_name="real_estate_data.xlsx",
        mime="application/vnd.ms-excel"
    )
    
    # レポート生成機能
    st.header('要約レポート')
    
    if st.button('要約レポートを生成'):
        report_text = generate_summary_report(filtered_df)
        st.markdown(report_text)
        
        # レポートのダウンロードボタン
        st.download_button(
            label="レポートをダウンロード",
            data=report_text,
            file_name="real_estate_report.md",
            mime="text/markdown",
        )
else:
    # データが読み込まれていない場合の表示
    st.info('データを取得するには、サイドバーで検索条件を設定し、「データ取得」ボタンをクリックしてください。')