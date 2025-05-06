# visualizers.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import japanize_matplotlib

def show_basic_stats(df):
    """基本的な統計情報を表示"""
    if 'TradePrice' not in df.columns or df['TradePrice'].isna().all():
        st.warning('価格データが存在しません')
        return
    
    # 単位を円として表示（変換はせず、フォーマットのみ変更）
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("平均価格", f"{df['TradePrice'].mean():,.0f}円")
        st.metric("最小価格", f"{df['TradePrice'].min():,.0f}円")
    
    with col2:
        st.metric("中央値", f"{df['TradePrice'].median():,.0f}円")
        st.metric("最大価格", f"{df['TradePrice'].max():,.0f}円")
    
    # 詳細な統計情報をテーブルとして表示
    st.write("詳細な統計情報:")
    
    # 統計量を計算
    count = len(df['TradePrice'].dropna())
    mean = df['TradePrice'].mean()
    std = df['TradePrice'].std()
    min_val = df['TradePrice'].min()
    q25 = df['TradePrice'].quantile(0.25)
    median = df['TradePrice'].median()
    q75 = df['TradePrice'].quantile(0.75)
    max_val = df['TradePrice'].max()
    
    # 統計情報をデータフレームとして整形
    stats_df = pd.DataFrame({
        '統計量': ['データ数', '平均値', '標準偏差', '最小値', '25%分位点', '中央値', '75%分位点', '最大値'],
        '価格（円）': [
            f"{count:.0f}",
            f"{mean:,.0f}",
            f"{std:,.0f}",
            f"{min_val:,.0f}",
            f"{q25:,.0f}",
            f"{median:,.0f}",
            f"{q75:,.0f}",
            f"{max_val:,.0f}"
        ]
    })
    
    st.dataframe(stats_df)

def plot_price_histogram(df):
    """価格分布のヒストグラム"""
    if 'TradePrice' not in df.columns or df['TradePrice'].isna().all():
        st.warning('価格データが存在しません')
        return
    
    fig = px.histogram(
        df, 
        x='TradePrice',
        nbins=30,
        title='取引価格の分布',
        labels={'TradePrice': '取引価格（万円）'},
        opacity=0.7
    )
    fig.update_layout(
        xaxis_title='取引価格（万円）',
        yaxis_title='件数'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_price_by_type(df):
    """タイプ別の価格分布"""
    if 'TradePrice' not in df.columns or 'Type' not in df.columns:
        st.warning('価格またはタイプデータが存在しません')
        return
    
    fig = px.box(
        df,
        x='Type',
        y='TradePrice',
        title='タイプ別の価格分布',
        labels={'TradePrice': '取引価格（万円）', 'Type': '物件タイプ'}
    )
    fig.update_layout(
        xaxis_title='物件タイプ',
        yaxis_title='取引価格（万円）'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_district_prices(df):
    """地区別の平均価格"""
    if 'TradePrice' not in df.columns or 'DistrictName' not in df.columns:
        st.warning('価格または地区データが存在しません')
        return
    
    # 地区別の平均価格と物件数を計算
    district_data = df.groupby('DistrictName').agg(
        平均価格=('TradePrice', 'mean'),
        物件数=('TradePrice', 'count')
    ).reset_index()
    
    district_data = district_data.sort_values('平均価格', ascending=False)
    
    # 上位10地区のみ表示（データが多い場合）
    if len(district_data) > 10:
        district_data = district_data.head(10)
        title = '平均価格上位10地区'
    else:
        title = '地区別の平均価格'
    
    fig = px.bar(
        district_data,
        x='DistrictName',
        y='平均価格',
        text='物件数',
        title=title,
        labels={'DistrictName': '地区名', '平均価格': '平均価格（万円）'}
    )
    fig.update_layout(
        xaxis_title='地区名',
        yaxis_title='平均価格（万円）'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_price_vs_building_age(df):
    """築年数と価格の関係"""
    if ('TradePrice' not in df.columns or 'BuildingAge' not in df.columns 
        or df['BuildingAge'].isna().all()):
        st.warning('価格または築年数データが存在しません')
        return
    
    # NaNを除外
    plot_df = df.dropna(subset=['TradePrice', 'BuildingAge'])
    
    fig = px.scatter(
        plot_df,
        x='BuildingAge',
        y='TradePrice',
        color='Type' if 'Type' in plot_df.columns else None,
        title='築年数と取引価格の関係',
        labels={
            'BuildingAge': '築年数（年）',
            'TradePrice': '取引価格（万円）',
            'Type': '物件タイプ'
        }
    )
    fig.update_layout(
        xaxis_title='築年数（年）',
        yaxis_title='取引価格（万円）'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_price_pie_chart(df):
    """価格帯別の割合（円グラフ）"""
    if 'PriceBin' not in df.columns:
        st.warning('価格帯データが存在しません')
        return
    
    count_by_bin = df['PriceBin'].value_counts().reset_index()
    count_by_bin.columns = ['価格帯', '件数']
    
    fig = px.pie(
        count_by_bin,
        values='件数',
        names='価格帯',
        title='価格帯別の割合'
    )
    st.plotly_chart(fig, use_container_width=True)
    
# visualizers.py に追加する関数

def plot_heatmap(df):
    """相関ヒートマップの表示"""
    # デバッグモードの確認
    debug_mode = st.session_state.get('debug_mode', False)
    
    # データのコピーを作成
    temp_df = df.copy()
    
    # 数値型のカラムを抽出
    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if debug_mode:
        st.write(f"元の数値カラム: {numeric_cols}")
    
    # カテゴリカル変数からダミー変数を作成
    if 'Type' in temp_df.columns and len(temp_df['Type'].unique()) <= 10:  # 種類が多すぎる場合は除外
        st.write("物件タイプのダミー変数を作成しています...")
        type_dummies = pd.get_dummies(temp_df['Type'], prefix='Type')
        temp_df = pd.concat([temp_df, type_dummies], axis=1)
    
    if 'DistrictName' in temp_df.columns:
        # 地区名は多すぎる可能性があるため、上位のみを使用
        top_districts = temp_df['DistrictName'].value_counts().head(5).index.tolist()
        st.write(f"上位5地区のダミー変数を作成しています: {', '.join(top_districts)}")
        
        # 上位地区のみでダミー変数を作成（プレフィックスなし）
        for district in top_districts:
            temp_df[district] = (temp_df['DistrictName'] == district).astype(int)
    
    # 数値型のカラムを再抽出（ダミー変数を含む）
    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if debug_mode:
        st.write(f"拡張後の数値カラム: {numeric_cols}")
    
    # 最低2つの数値カラムが必要
    if len(numeric_cols) < 2:
        st.warning('相関分析に必要な数値カラムが不足しています')
        return
    
    # NaNが含まれる場合は相関を計算できない列を除外
    numeric_cols = [col for col in numeric_cols if not temp_df[col].isna().all()]
    if len(numeric_cols) < 2:
        st.warning('有効な数値データが不足しています（すべてNaN）')
        return
    
    # 相関係数の計算
    corr = temp_df[numeric_cols].corr()
    
    # ヒートマップの作成
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 上三角マスク（重複を避ける）
    
    # ブルー系統のカラーマップを使用
    sns.heatmap(corr, mask=mask, annot=True, cmap='Blues', fmt='.2f', ax=ax, linewidths=.5)
    ax.set_title('数値変数間の相関係数', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 軸ラベルを見やすく調整
    plt.tight_layout()
    st.pyplot(fig)
    
    # 興味深い相関関係の解説
    st.subheader("相関関係の解説")
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.5:  # 相関係数の絶対値が0.5以上
                var1 = corr.columns[i]
                var2 = corr.columns[j]
                coef = corr.iloc[i, j]
                high_corr.append((var1, var2, coef))
    
    if high_corr:
        st.write("強い相関関係（相関係数の絶対値が0.5以上）があるペア:")
        for var1, var2, coef in high_corr:
            direction = "正の" if coef > 0 else "負の"
            strength = "非常に強い" if abs(coef) > 0.8 else "強い"
            
            # 変数名をよりユーザーフレンドリーに表示
            display_var1 = var1.replace('District_', '')
            display_var2 = var2.replace('District_', '')
            
            st.write(f"- **{display_var1}** と **{display_var2}** の間に{strength}{direction}相関 ({coef:.2f}) があります。")
    else:
        st.write("強い相関関係を持つ変数ペアは見つかりませんでした。")

def plot_time_trend(df):
    """時系列トレンドの分析"""
    if 'Period' not in df.columns or 'TradePrice' not in df.columns:
        st.warning('時系列分析に必要なデータが不足しています')
        return
    
    # 期間ごとの平均価格を計算
    period_price = df.groupby('Period')['TradePrice'].agg(['mean', 'count']).reset_index()
    period_price = period_price.sort_values('Period')
    
    # ブルー系統のカラーパレット
    primary_blue = 'rgba(0, 123, 255, 0.7)'    # メインの青色（棒グラフ）
    secondary_blue = 'rgba(30, 67, 132, 0.9)'  # 濃い青色（線グラフ）
    
    # Plotlyを使った複合グラフの作成
    fig = go.Figure()
    
    # 棒グラフ（取引件数）
    fig.add_trace(go.Bar(
        x=period_price['Period'],
        y=period_price['count'],
        name='取引件数',
        marker_color=primary_blue,
        opacity=0.7
    ))
    
    # 線グラフ（平均価格）- 右軸に配置
    fig.add_trace(go.Scatter(
        x=period_price['Period'],
        y=period_price['mean'],
        name='平均価格（円）',
        mode='lines+markers',
        marker=dict(size=10, color=secondary_blue),
        line=dict(width=3, color=secondary_blue),
        yaxis='y2'
    ))
    
    # レイアウト設定
    fig.update_layout(
        title='期間ごとの平均価格と取引件数の推移',
        xaxis=dict(
            title='期間',
            tickangle=-45,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='取引件数',
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            title='平均価格（円）',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=50, r=50, t=80, b=80),
        height=600,
        hovermode='x unified',
        template='plotly_white'  # クリーンなテンプレート
    )
    
    # Y軸のグリッド線を棒グラフにのみ表示
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', griddash='dot')
    
    # 棒グラフと線グラフの間隔を調整
    fig.update_traces(
        width=0.6,
        selector=dict(type='bar')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 時系列トレンドの解説
    if len(period_price) > 1:
        # 最初と最後の期間を取得
        first_period = period_price.iloc[0]
        last_period = period_price.iloc[-1]
        
        # 平均価格の変化率を計算
        price_change_pct = ((last_period['mean'] - first_period['mean']) / first_period['mean']) * 100
        
        # 取引件数の変化率を計算
        count_change_pct = ((last_period['count'] - first_period['count']) / first_period['count']) * 100
        
        st.write("### 時系列トレンドの解説")
        
        price_trend = "上昇" if price_change_pct > 0 else "下降"
        count_trend = "増加" if count_change_pct > 0 else "減少"
        
        st.write(f"- {first_period['Period']}から{last_period['Period']}にかけて、平均価格は{abs(price_change_pct):.1f}%の{price_trend}傾向を示しています。")
        st.write(f"- 同期間で、取引件数は{abs(count_change_pct):.1f}%の{count_trend}傾向を示しています。")
    else:
        st.write("時系列分析には複数の期間のデータが必要です。現在は単一期間のデータのみが存在します。")

def plot_price_per_sqm(df):
    """平米単価の分析"""
    if 'PricePerSquareMeter' not in df.columns or df['PricePerSquareMeter'].isna().all():
        st.warning('平米単価データが存在しません')
        return
    
    # 平米単価のヒストグラム
    fig = px.histogram(
        df.dropna(subset=['PricePerSquareMeter']),
        x='PricePerSquareMeter',
        nbins=30,
        title='平米単価の分布',
        labels={'PricePerSquareMeter': '平米単価（万円/㎡）'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 物件タイプ別の平米単価
    if 'Type' in df.columns:
        fig = px.box(
            df.dropna(subset=['PricePerSquareMeter', 'Type']),
            x='Type',
            y='PricePerSquareMeter',
            title='物件タイプ別の平米単価',
            labels={'Type': '物件タイプ', 'PricePerSquareMeter': '平米単価（万円/㎡）'}
        )
        st.plotly_chart(fig, use_container_width=True)