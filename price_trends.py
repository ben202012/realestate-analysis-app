# price_trends.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import japanize_matplotlib
from data_handler import fetch_and_process_data


def fetch_historical_price_data(api_key, prefecture_code, city_code, base_year=None, property_type=None, quarters=8):
    """
    過去n四半期分のデータを取得する関数
    
    Parameters:
    -----------
    api_key : str
        APIキー
    prefecture_code : int
        都道府県コード
    city_code : int
        市区町村コード
    base_year : str or int, optional
        基準年（Noneの場合は現在年を使用）
    property_type : str, optional
        物件タイプ（フィルタリング用）
    quarters : int, optional
        取得する四半期の数（デフォルト: 8 = 2年分）
    
    Returns:
    --------
    pandas.DataFrame
        四半期ごとのデータを含むDataFrame
    """
    # 基準年と四半期を設定
    if base_year is None:
        # 基準年が指定されていない場合は現在の年を使用
        current_date = datetime.now()
        base_year = current_date.year
        current_quarter = (current_date.month - 1) // 3 + 1
    else:
        # 基準年が指定されている場合は、その年の第4四半期を基準にする
        base_year = int(base_year)
        current_quarter = 4
    
    # 過去n四半期分のデータを格納するリスト
    all_data = []
    
    # 取得成功・失敗の四半期を記録するリスト
    success_quarters = []
    failed_quarters = []
    
    # プログレスバーを表示
    progress_bar = st.progress(0)
    
    # 取得する四半期の一覧を作成
    quarters_to_fetch = []
    year = base_year
    quarter = current_quarter
    
    for i in range(quarters):
        quarters_to_fetch.append((year, quarter))
        quarter -= 1
        if quarter < 1:
            quarter = 4
            year -= 1
    
    # 四半期データを取得
    for i, (year, quarter) in enumerate(quarters_to_fetch):
        # プログレスバーを更新
        progress_bar.progress((i + 1) / len(quarters_to_fetch))
        
        # データ取得
        quarter_label = f'{year}年 第{quarter}四半期'
        with st.spinner(f'{quarter_label}のデータを取得中...'):
            try:
                df = fetch_and_process_data(api_key, str(year), prefecture_code, city_code)
                
                if df is not None and not df.empty:
                    # 四半期情報を追加
                    df['Year'] = year
                    df['Quarter'] = quarter
                    df['YearQuarter'] = f'{year}Q{quarter}'
                    
                    # 物件タイプでフィルタリング（指定がある場合）
                    if property_type and 'Type' in df.columns:
                        df = df[df['Type'] == property_type]
                    
                    all_data.append(df)
                    st.session_state[f'data_{year}_q{quarter}'] = df
                    success_quarters.append(quarter_label)
                else:
                    # データが空または取得できなかった場合
                    failed_quarters.append(quarter_label)
            except Exception as e:
                # エラーが発生した場合
                failed_quarters.append(quarter_label)
                if st.session_state.get('debug_mode', False):
                    st.error(f'{quarter_label}のデータ取得中にエラーが発生しました: {str(e)}')
    
    # プログレスバーを完了状態に
    progress_bar.progress(1.0)
    progress_bar.empty()
    
    # 成功・失敗の四半期をまとめて表示
    if success_quarters:
        st.success(f"データ取得成功: {len(success_quarters)}四半期 ({', '.join(success_quarters)})")
    
    if failed_quarters:
        # デバッグモードでない場合は警告として表示
        message = f"データが存在しない四半期: {len(failed_quarters)}四半期 ({', '.join(failed_quarters)})"
        if st.session_state.get('debug_mode', False):
            st.error(message)
        else:
            st.warning(message)
    
    # すべてのデータを結合
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    return None


def extract_quarter_from_period(period_str):
    """
    期間文字列から四半期情報を抽出する関数
    
    Parameters:
    -----------
    period_str : str
        期間文字列（例: '2023年第３四半期'）
    
    Returns:
    --------
    tuple
        (年, 四半期) の形式で返す
    """
    if not isinstance(period_str, str):
        return None
        
    # 正規表現で年と四半期を抽出
    match = re.search(r'(\d{4})年第(\d)四半期', period_str)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        return (year, quarter)
    return None


def prepare_quarterly_price_data(df):
    """
    四半期ごとの価格データを準備する関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        取得したデータ
    
    Returns:
    --------
    pandas.DataFrame
        四半期ごとの集計データ
    """
    if df is None or df.empty:
        return None
    
    # 四半期情報がない場合は'Period'から抽出
    if 'Quarter' not in df.columns and 'Period' in df.columns:
        # 期間から年と四半期を抽出
        quarter_info = df['Period'].apply(extract_quarter_from_period)
        df['Year'] = quarter_info.apply(lambda x: x[0] if x else None)
        df['Quarter'] = quarter_info.apply(lambda x: x[1] if x else None)
        df['YearQuarter'] = df.apply(lambda row: f"{row['Year']}Q{row['Quarter']}" if pd.notna(row['Year']) and pd.notna(row['Quarter']) else None, axis=1)
    
    # 物件タイプと四半期ごとの集計
    if 'Type' in df.columns and 'TradePrice' in df.columns and 'YearQuarter' in df.columns:
        # 欠損値を除外
        valid_data = df.dropna(subset=['Type', 'TradePrice', 'YearQuarter'])
        
        # タイプと四半期ごとの平均価格を計算
        quarterly_prices = valid_data.groupby(['Type', 'Year', 'Quarter', 'YearQuarter'])['TradePrice'].agg(['mean', 'count', 'std']).reset_index()
        
        # 平均価格を万円単位に変換
        quarterly_prices['mean_man'] = quarterly_prices['mean'] / 10000
        
        # 標準偏差を万円単位に変換（stdがNaNの場合は0にする）
        if 'std' in quarterly_prices.columns:
            quarterly_prices['std_man'] = quarterly_prices['std'].fillna(0) / 10000
        else:
            quarterly_prices['std_man'] = 0
        
        # 四半期の時系列順でソート
        quarterly_prices = quarterly_prices.sort_values(['Type', 'Year', 'Quarter'])
        
        return quarterly_prices
    
    return None


def plot_quarterly_price_trends(quarterly_prices, property_types=None):
    """
    四半期ごとの価格推移をプロットする関数
    
    Parameters:
    -----------
    quarterly_prices : pandas.DataFrame
        四半期ごとの集計データ
    property_types : list, optional
        表示する物件タイプのリスト（Noneの場合はすべて表示）
    
    Returns:
    --------
    None
    """
    if quarterly_prices is None or quarterly_prices.empty:
        st.warning('集計に必要なデータが不足しています')
        return
    
    # 表示する物件タイプを制限（指定がある場合）
    if property_types:
        quarterly_prices = quarterly_prices[quarterly_prices['Type'].isin(property_types)]
    
    if quarterly_prices.empty:
        st.warning('選択された物件タイプのデータが存在しません')
        return
    
    # プロット用のカラーマップを定義
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'teal', 'pink', 'brown']
    
    # Plotlyを使用した折れ線グラフの作成
    fig = go.Figure()
    
    # 各物件タイプに対して折れ線を追加
    for i, prop_type in enumerate(quarterly_prices['Type'].unique()):
        type_data = quarterly_prices[quarterly_prices['Type'] == prop_type]
        
        # 色のインデックス（物件タイプの数が多い場合はループする）
        color_idx = i % len(colors)
        
        # メインの色取得
        main_color = colors[color_idx]
        # 透明度付きの色（塗りつぶし用）
        fill_color = f'rgba(0,0,250,0.2)' if main_color == 'blue' else \
                    f'rgba(250,0,0,0.2)' if main_color == 'red' else \
                    f'rgba(0,150,0,0.2)' if main_color == 'green' else \
                    f'rgba(150,0,150,0.2)' if main_color == 'purple' else \
                    f'rgba(250,150,0,0.2)' if main_color == 'orange' else \
                    f'rgba(0,150,150,0.2)' if main_color == 'teal' else \
                    f'rgba(250,150,150,0.2)' if main_color == 'pink' else \
                    f'rgba(150,75,0,0.2)'  # brown
        
        # 折れ線グラフの追加
        fig.add_trace(go.Scatter(
            x=type_data['YearQuarter'],
            y=type_data['mean_man'],
            name=prop_type,
            mode='lines+markers',
            line=dict(color=main_color, width=3),
            marker=dict(size=10, color=main_color),
            hovertemplate='%{y:.1f}万円<br>取引件数: %{text}<extra></extra>',
            text=type_data['count']
        ))
        
        # 標準偏差の範囲を示す塗りつぶし（オプション）
        if 'std_man' in type_data.columns and not type_data['std_man'].isnull().all():
            upper_bounds = type_data['mean_man'] + type_data['std_man']
            lower_bounds = type_data['mean_man'] - type_data['std_man']
            
            # マイナスにならないように調整
            lower_bounds = lower_bounds.apply(lambda x: max(0, x))
            
            fig.add_trace(go.Scatter(
                x=type_data['YearQuarter'],
                y=upper_bounds,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=type_data['YearQuarter'],
                y=lower_bounds,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # グラフのレイアウト設定
    fig.update_layout(
        title='四半期ごとの平均価格推移',
        xaxis_title='四半期',
        yaxis_title='平均価格（万円）',
        hovermode='x unified',
        legend_title='物件タイプ',
        template='plotly_white',
        height=600,
        margin=dict(l=50, r=50, t=60, b=80)
    )
    
    # X軸のフォーマット設定
    fig.update_xaxes(
        tickangle=-45,
        tickfont=dict(size=12),
        tickmode='array',
        tickvals=quarterly_prices['YearQuarter'].unique()
    )
    
    # Y軸のグリッド線を設定
    fig.update_yaxes(
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        griddash='dot'
    )
    
    # グラフを表示
    st.plotly_chart(fig, use_container_width=True)
    
    # グラフの説明を追加
    with st.expander("📊 グラフの見方"):
        st.markdown("""
        ### 四半期ごとの平均価格推移グラフの見方
        
        このグラフは各四半期における物件タイプ別の平均価格推移を表示しています。
        
        #### 主な表示要素:
        
        - **実線**: 各四半期の平均価格を示しています。
        - **薄い色の帯**: 平均価格の周りの**標準偏差**の範囲を示しています。
        
        #### 標準偏差とは？
        
        標準偏差は価格のばらつき具合を数値化したものです。
        
        - **標準偏差が大きい（帯が広い）場合**: その四半期の物件価格には大きなばらつきがあります。これは高額物件と低額物件の両方が混在していることを意味します。
        
        - **標準偏差が小さい（帯が狭い）場合**: 物件価格が平均値の周りに集中しており、価格が比較的均一であることを示します。
        
        #### 分析のヒント:
        
        - **帯が急に広がる**: 市場の価格差が拡大し、価格が多様化している可能性があります。
        - **帯が徐々に狭まる**: 市場が標準化され、価格が収束する傾向にあります。
        - **物件タイプによる違い**: 一部の物件タイプは他より標準偏差が大きい傾向があり、価格のばらつきに差があります。
        
        マウスをグラフ上に置くと、その四半期の平均価格と取引件数が表示されます。
        """)
    
    # トレンド分析のサマリーを表示
    show_trend_analysis(quarterly_prices)


def show_trend_analysis(quarterly_prices):
    """
    価格トレンドの分析サマリーを表示する関数
    
    Parameters:
    -----------
    quarterly_prices : pandas.DataFrame
        四半期ごとの集計データ
    
    Returns:
    --------
    None
    """
    if quarterly_prices is None or quarterly_prices.empty:
        return
    
    st.subheader('価格トレンド分析')
    
    # 物件タイプごとに分析
    for prop_type in quarterly_prices['Type'].unique():
        type_data = quarterly_prices[quarterly_prices['Type'] == prop_type]
        
        # 2四半期以上のデータがある場合のみトレンド分析
        if len(type_data) >= 2:
            # 最初と最後の四半期を取得
            first_quarter = type_data.iloc[0]
            last_quarter = type_data.iloc[-1]
            
            # 価格変化率を計算
            price_change = (last_quarter['mean'] - first_quarter['mean']) / first_quarter['mean'] * 100
            
            # トレンドの方向を決定
            trend_direction = "上昇" if price_change > 0 else "下降"
            
            # 結果を表示
            st.markdown(f"""
            ### {prop_type}
            - {first_quarter['YearQuarter']} から {last_quarter['YearQuarter']} にかけて平均価格は **{abs(price_change):.1f}%** の**{trend_direction}**傾向
            - 四半期あたりの平均取引件数: **{type_data['count'].mean():.1f}件**
            """)
            
            # 直近のトレンドを分析（直近3四半期のデータがある場合）
            if len(type_data) >= 3:
                recent_data = type_data.iloc[-3:]
                
                # 直近の価格変化率を計算
                recent_change = (recent_data.iloc[-1]['mean'] - recent_data.iloc[0]['mean']) / recent_data.iloc[0]['mean'] * 100
                
                # 直近のトレンドと全体のトレンドを比較
                if (recent_change > 0) != (price_change > 0):
                    recent_trend = "上昇" if recent_change > 0 else "下降"
                    st.markdown(f"- **直近のトレンド**: 全体とは異なり、最近の3四半期では **{recent_trend}**傾向 ({abs(recent_change):.1f}%)")
                elif abs(recent_change) > abs(price_change) / len(type_data) * 3 * 1.5:
                    st.markdown(f"- **直近のトレンド**: 全体よりも**急激な{trend_direction}**傾向 ({abs(recent_change):.1f}%)")
        else:
            st.markdown(f"### {prop_type}\n- トレンド分析には少なくとも2四半期分のデータが必要です")


def show_quarterly_price_trends(api_key, prefecture_code, city_code, year=None):
    """
    四半期ごとの価格推移を表示するメイン関数
    
    Parameters:
    -----------
    api_key : str
        APIキー
    prefecture_code : int
        都道府県コード
    city_code : int
        市区町村コード
    year : str, optional
        基準年（選択された年）
    
    Returns:
    --------
    None
    """
    st.header('四半期ごとの価格推移')
    
    # サイドバーに設定オプションを表示
    with st.sidebar.expander("価格推移の設定"):
        # 取得する四半期数の設定
        quarters = st.slider('取得する四半期数', min_value=4, max_value=12, value=8, step=1, 
                            help='過去何四半期分のデータを取得するか（8 = 2年分）')
        
        # 基準年の選択（デフォルトは現在選択されている年）
        available_years = ['2024', '2023', '2022', '2021', '2020', '2019']
        if year is not None and year in available_years:
            default_year_index = available_years.index(year)
        else:
            default_year_index = 0
            
        base_year = st.selectbox(
            '基準年',
            options=available_years,
            index=default_year_index,
            help='この年の第4四半期から遡ってデータを取得します'
        )
        
        # 物件タイプの読み込みと選択肢の準備
        if 'data' in st.session_state:
            df = st.session_state['data']
            if 'Type' in df.columns:
                available_types = df['Type'].dropna().unique().tolist()
                selected_types = st.multiselect(
                    '表示する物件タイプ', 
                    options=available_types,
                    default=available_types[:3] if len(available_types) > 0 else [],
                    help='グラフに表示する物件タイプを選択（未選択の場合はすべて表示）'
                )
            else:
                selected_types = None
        else:
            selected_types = None
    
    # 価格推移データの取得ボタン
    if st.button('四半期ごとの価格推移を取得'):
        if api_key:
            with st.spinner('過去の価格データを取得中...'):
                # 過去のデータを取得（基準年を指定）
                historical_data = fetch_historical_price_data(
                    api_key, prefecture_code, city_code, 
                    base_year=base_year,  # 基準年を渡す
                    property_type=None, 
                    quarters=quarters
                )
                
                if historical_data is not None and not historical_data.empty:
                    # 四半期ごとの価格データを準備
                    quarterly_prices = prepare_quarterly_price_data(historical_data)
                    
                    if quarterly_prices is not None and not quarterly_prices.empty:
                        # セッションに保存
                        st.session_state['quarterly_prices'] = quarterly_prices
                        
                        # データの概要を表示
                        st.info(f"四半期データ集計完了: {len(quarterly_prices)}件の物件タイプ別・四半期別データ")
                        
                        # 四半期ごとの価格推移をプロット
                        plot_quarterly_price_trends(quarterly_prices, selected_types)
                    else:
                        st.error('四半期ごとの価格データの準備に失敗しました')
                else:
                    st.error('過去のデータ取得に失敗しました')
        else:
            st.error('APIキーが設定されていません')
    
    # セッションに四半期価格データがあれば表示
    elif 'quarterly_prices' in st.session_state:
        plot_quarterly_price_trends(st.session_state['quarterly_prices'], selected_types)


# データテーブル表示（デバッグ用）
def show_quarterly_data_table():
    """
    四半期ごとのデータテーブルを表示する関数（デバッグ用）
    """
    if 'quarterly_prices' in st.session_state:
        with st.expander('四半期ごとの価格データ（詳細）'):
            st.dataframe(st.session_state['quarterly_prices'])