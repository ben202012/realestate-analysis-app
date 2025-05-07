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
from models import train_price_prediction_model, show_feature_importance, predict_price
from utils import get_prefecture_dict, get_city_options, format_price, filter_dataframe, generate_summary_report

# タブ選択の状態を保持するための関数
def train_model_on_click(df, features):
    with st.spinner('モデルを構築中...'):
        model, mse, r2 = train_price_prediction_model(df, features)
        if model is not None:
            st.session_state['model_r2'] = r2
            st.session_state['model_mse'] = mse

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

# アプリのタイトルと説明
st.title('不動産取引価格分析アプリ')
st.write('国土交通省APIから取得した不動産取引データを分析・可視化するアプリケーションです')

# サイドバー - 検索条件
st.sidebar.header('検索条件')

# 年度選択
year = st.sidebar.selectbox(
    '年度',
    options=['2024', '2023', '2022', '2021', '2020', '2019'],
    index=0  # 2024年をデフォルト選択
)

# 都道府県選択
prefecture_dict = get_prefecture_dict()
prefecture_code = st.sidebar.selectbox(
    '都道府県',
    options=list(prefecture_dict.keys()),
    format_func=lambda x: prefecture_dict[x],
    index=list(prefecture_dict.keys()).index(27)  # 大阪府をデフォルト選択
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
                    index=0
                )
            else:
                # API取得に失敗した場合、旧方式で表示
                city_options = get_city_options(prefecture_code)
                city_code = st.selectbox(
                    '市区町村',
                    options=[opt[0] for opt in city_options],
                    format_func=lambda x: dict(city_options)[x],
                    index=0
                )
                st.warning("APIから市区町村リストを取得できなかったため、一部の市区町村のみ表示しています。")
else:
    # APIキーが未入力の場合は従来の方法でリスト表示
    city_options = get_city_options(prefecture_code)
    city_code = st.sidebar.selectbox(
        '市区町村',
        options=[opt[0] for opt in city_options],
        format_func=lambda x: dict(city_options)[x],
        index=0
    )

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

# サイドバー - 開発者モード
st.sidebar.header('開発者オプション')
# セッション状態にデバッグモードが存在しない場合は初期化
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False
# チェックボックスの値をセッション状態に反映
debug_mode = st.sidebar.checkbox('デバッグモードを有効にする', value=st.session_state['debug_mode'])
st.session_state['debug_mode'] = debug_mode

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
    
    # フィルタリング適用
    filtered_df = filter_dataframe(df, filters)
    
    st.sidebar.write(f'フィルタリング結果: {len(filtered_df)}件 / 全{len(df)}件')
else:
    filtered_df = None

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
    tab_names = ['基本統計', '価格分析', 'エリア分析', '高度な分析', '価格予測']
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
        st.write('機械学習を使った簡易的な価格予測モデルを構築します。')
        
        # 特徴量選択
        available_features = ['Area', 'BuildingAge', 'Type', 'Renovation']
        selected_features = st.multiselect(
            '使用する特徴量を選択',
            options=[f for f in available_features if f in filtered_df.columns],
            default=[f for f in ['Area', 'BuildingAge', 'Type'] if f in filtered_df.columns]
        )
        
        # フォームを使用してモデル構築ボタンをグループ化
        with st.form(key='model_build_form'):
            submit_build = st.form_submit_button('モデル構築')
            
        if submit_build and len(selected_features) > 0:
            train_model_on_click(filtered_df, selected_features)
        
        # 特徴量の重要度
        if 'prediction_model' in st.session_state and 'model_features' in st.session_state:
            show_feature_importance(st.session_state['prediction_model'], st.session_state['model_features'])
        
        # 予測フォームと結果の表示
        from models import show_prediction_form_and_results
        show_prediction_form_and_results(filtered_df)

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