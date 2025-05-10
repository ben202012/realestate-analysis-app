# utils.py
import pandas as pd
import streamlit as st

def get_prefecture_dict():
    """都道府県コードと名称の辞書を返す"""
    prefectures = [
        (1, '北海道'), (2, '青森県'), (3, '岩手県'), (4, '宮城県'), (5, '秋田県'),
        (6, '山形県'), (7, '福島県'), (8, '茨城県'), (9, '栃木県'), (10, '群馬県'),
        (11, '埼玉県'), (12, '千葉県'), (13, '東京都'), (14, '神奈川県'), (15, '新潟県'),
        (16, '富山県'), (17, '石川県'), (18, '福井県'), (19, '山梨県'), (20, '長野県'),
        (21, '岐阜県'), (22, '静岡県'), (23, '愛知県'), (24, '三重県'), (25, '滋賀県'),
        (26, '京都府'), (27, '大阪府'), (28, '兵庫県'), (29, '奈良県'), (30, '和歌山県'),
        (31, '鳥取県'), (32, '島根県'), (33, '岡山県'), (34, '広島県'), (35, '山口県'),
        (36, '徳島県'), (37, '香川県'), (38, '愛媛県'), (39, '高知県'), (40, '福岡県'),
        (41, '佐賀県'), (42, '長崎県'), (43, '熊本県'), (44, '大分県'), (45, '宮崎県'),
        (46, '鹿児島県'), (47, '沖縄県')
    ]
    return dict(prefectures)

def get_city_options(prefecture_code):
    """都道府県に属する市区町村の選択肢（ダミーデータ、実際にはAPIから取得するか事前に用意）"""
    # 大阪府の場合の例
    if prefecture_code == 27:
        return [
            (27100, '大阪市'),
            (27140, '堺市'),
            (27202, '岸和田市'),
            (27203, '豊中市'),
            (27204, '池田市'),
            (27205, '吹田市'),
            (27206, '泉大津市'),
            (27207, '高槻市'),
            (27208, '貝塚市'),
            (27209, '守口市'),
            (27210, '枚方市')
        ]
    # 東京都の場合の例
    elif prefecture_code == 13:
        return [
            (13101, '千代田区'),
            (13102, '中央区'),
            (13103, '港区'),
            (13104, '新宿区'),
            (13105, '文京区'),
            (13106, '台東区'),
            (13107, '墨田区'),
            (13108, '江東区'),
            (13109, '品川区'),
            (13110, '目黒区')
        ]
    # その他の都道府県の場合はダミーデータ
    else:
        return [(prefecture_code * 1000 + 1, 'サンプル市')]

def format_price(price):
    """価格を見やすい形式にフォーマット（万円単位）"""
    price_man = price / 10000  # 円から万円に変換
    return f"{price_man:,.1f}万円"

def filter_dataframe(df, filters):
    """条件によるデータのフィルタリング"""
    # デバッグモードの確認
    debug_mode = st.session_state.get('debug_mode', False)
    
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    # TradePrice列が文字列の場合、数値に変換
    if 'TradePrice' in filtered_df.columns:
        filtered_df['TradePrice'] = pd.to_numeric(filtered_df['TradePrice'], errors='coerce')
        if debug_mode:
            st.write(f"価格データ型変換後: {len(filtered_df[(~filtered_df['TradePrice'].isna())])}件")
    
    # 価格範囲によるフィルタリング
    if 'price_range' in filters and 'TradePrice' in filtered_df.columns:
        min_price, max_price = filters['price_range']
        if debug_mode:
            st.write(f"価格範囲: {min_price:,}円 〜 {max_price:,}円")
        
        # フィルタリング前後の件数を確認
        before_filter = len(filtered_df)
        filtered_df = filtered_df[(filtered_df['TradePrice'] >= min_price) & 
                                (filtered_df['TradePrice'] <= max_price)]
        after_filter = len(filtered_df)
        if debug_mode:
            st.write(f"価格フィルタ適用: {before_filter}件 → {after_filter}件")
    
    # 物件タイプによるフィルタリング
    if 'property_type' in filters and filters['property_type'] != 'すべて' and 'Type' in filtered_df.columns:
        before_filter = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Type'] == filters['property_type']]
        after_filter = len(filtered_df)
        if debug_mode:
            st.write(f"物件タイプフィルタ適用: {before_filter}件 → {after_filter}件")
    
    # 築年数によるフィルタリング
    if 'building_age_range' in filters and 'BuildingAge' in filtered_df.columns:
        min_age, max_age = filters['building_age_range']
        # BuildingAge列が文字列の場合、数値に変換
        if filtered_df['BuildingAge'].dtype == 'object':
            filtered_df['BuildingAge'] = pd.to_numeric(filtered_df['BuildingAge'], errors='coerce')
        
        before_filter = len(filtered_df)
        filtered_df = filtered_df[(filtered_df['BuildingAge'] >= min_age) & 
                                (filtered_df['BuildingAge'] <= max_age)]
        after_filter = len(filtered_df)
        if debug_mode:
            st.write(f"築年数フィルタ適用: {before_filter}件 → {after_filter}件")
    
    final_count = len(filtered_df)
    if debug_mode:
        st.write(f"フィルタリング合計: {original_count}件 → {final_count}件")
    
    return filtered_df

def format_stats_value(value, is_price=True):
    """統計値を適切な形式にフォーマット"""
    if pd.isna(value):
        return "N/A"
    
    if isinstance(value, (int, float)):
        if value == int(value):
            # 整数の場合
            formatted = f"{int(value):,}"
        else:
            # 小数の場合、小数点以下1桁で表示
            formatted = f"{value:.1f}"
        
        if is_price:
            return f"{formatted}万円"
        return formatted
    
    return str(value)

# utils.py 内の generate_summary_report 関数の改善部分

def generate_summary_report(df):
    """データの要約レポートを生成"""
    if df is None or df.empty:
        return "データがありません。"
    
    # 基本情報
    report = "# 不動産取引データ 要約レポート\n\n"
    report += f"## 基本情報\n"
    report += f"- データ件数: {len(df):,}件\n"
    
    if 'Prefecture' in df.columns:
        prefecture = df['Prefecture'].iloc[0] if not df['Prefecture'].isna().all() else "不明"
        report += f"- 都道府県: {prefecture}\n"
    
    if 'Municipality' in df.columns:
        municipality = df['Municipality'].iloc[0] if not df['Municipality'].isna().all() else "不明"
        report += f"- 市区町村: {municipality}\n"
    
    if 'Period' in df.columns:
        periods = df['Period'].unique()
        report += f"- 対象期間: {', '.join(sorted(periods))}\n"
    
    # 価格統計
    if 'TradePrice' in df.columns:
        report += f"\n## 価格統計\n"
        report += f"- 平均価格: {df['TradePrice'].mean()/10000:,.1f}万円\n"
        report += f"- 中央値: {df['TradePrice'].median()/10000:,.1f}万円\n"
        report += f"- 最小値: {df['TradePrice'].min()/10000:,.1f}万円\n"
        report += f"- 最大値: {df['TradePrice'].max()/10000:,.1f}万円\n"
        report += f"- 標準偏差: {df['TradePrice'].std()/10000:,.1f}万円\n"
        
        # 四分位数も追加
        report += f"- 第1四分位数 (25%): {df['TradePrice'].quantile(0.25)/10000:,.1f}万円\n"
        report += f"- 第3四分位数 (75%): {df['TradePrice'].quantile(0.75)/10000:,.1f}万円\n"
        
        # 単価の基本統計（あれば）
        if 'PricePerSquareMeter' in df.columns:
            valid_unit_price = df['PricePerSquareMeter'].dropna()
            if not valid_unit_price.empty:
                report += f"\n### 平米単価の統計\n"
                report += f"- 平均単価: {valid_unit_price.mean():,.1f}万円/㎡\n"
                report += f"- 中央値: {valid_unit_price.median():,.1f}万円/㎡\n"
                report += f"- 標準偏差: {valid_unit_price.std():,.1f}万円/㎡\n"
    
    # 物件タイプ別の分布
    if 'Type' in df.columns:
        report += f"\n## 物件タイプ別の分布\n"
        type_counts = df['Type'].value_counts()
        
        # 物件タイプ別の価格統計も追加
        if 'TradePrice' in df.columns:
            report += "### 物件タイプ別の価格統計（万円）\n"
            report += "| 物件タイプ | 件数 | 割合 | 平均価格 | 中央値 | 最小値 | 最大値 |\n"
            report += "|------------|------|------|----------|--------|--------|--------|\n"
            
            for type_name, count in type_counts.items():
                percentage = 100 * count / len(df)
                type_df = df[df['Type'] == type_name]
                
                avg_price = type_df['TradePrice'].mean() / 10000
                median_price = type_df['TradePrice'].median() / 10000
                min_price = type_df['TradePrice'].min() / 10000
                max_price = type_df['TradePrice'].max() / 10000
                
                report += f"| {type_name} | {count:,} | {percentage:.1f}% | {avg_price:,.1f} | {median_price:,.1f} | {min_price:,.1f} | {max_price:,.1f} |\n"
        else:
            # 物件タイプの単純な分布
            for type_name, count in type_counts.items():
                percentage = 100 * count / len(df)
                report += f"- {type_name}: {count}件 ({percentage:.1f}%)\n"
    
    # 価格帯別の分布
    if 'PriceBin' in df.columns:
        report += f"\n## 価格帯別の分布\n"
        price_bin_counts = df['PriceBin'].value_counts().sort_index()
        
        for bin_name, count in price_bin_counts.items():
            percentage = 100 * count / len(df)
            report += f"- {bin_name}: {count}件 ({percentage:.1f}%)\n"
    
    # 地区別の平均価格（上位5地区）
    if 'DistrictName' in df.columns and 'TradePrice' in df.columns:
        report += f"\n## 地区別の分析\n"
        district_price = df.groupby('DistrictName')['TradePrice'].agg(['mean', 'count', 'median', 'min', 'max']).reset_index()
        district_price = district_price.sort_values('mean', ascending=False)
        
        report += "### 平均価格上位5地区（万円）\n"
        report += "| 地区名 | 平均価格 | 中央値 | 最小値 | 最大値 | 件数 |\n"
        report += "|--------|----------|--------|--------|--------|------|\n"
        
        for _, row in district_price.head(5).iterrows():
            report += f"| {row['DistrictName']} | {row['mean']/10000:,.1f} | {row['median']/10000:,.1f} | {row['min']/10000:,.1f} | {row['max']/10000:,.1f} | {row['count']:,} |\n"
        
        # 平均価格下位5地区も追加
        if len(district_price) > 10:
            report += "\n### 平均価格下位5地区（万円）\n"
            report += "| 地区名 | 平均価格 | 中央値 | 最小値 | 最大値 | 件数 |\n"
            report += "|--------|----------|--------|--------|--------|------|\n"
            
            for _, row in district_price.tail(5).iterrows():
                report += f"| {row['DistrictName']} | {row['mean']/10000:,.1f} | {row['median']/10000:,.1f} | {row['min']/10000:,.1f} | {row['max']/10000:,.1f} | {row['count']:,} |\n"
    
    # 築年数の統計（ある場合）
    if 'BuildingAge' in df.columns:
        report += f"\n## 築年数の統計\n"
        report += f"- 平均築年数: {df['BuildingAge'].mean():.1f}年\n"
        report += f"- 中央値: {df['BuildingAge'].median():.1f}年\n"
        report += f"- 最小値: {df['BuildingAge'].min():.1f}年\n"
        report += f"- 最大値: {df['BuildingAge'].max():.1f}年\n"
        
        # 築年数別の価格分析も追加
        if 'TradePrice' in df.columns:
            # 築年数を5年ごとの区間に分類
            df['AgeGroup'] = pd.cut(df['BuildingAge'], 
                                   bins=[0, 5, 10, 15, 20, 25, 30, 100], 
                                   labels=['0-5年', '6-10年', '11-15年', '16-20年', '21-25年', '26-30年', '31年以上'])
            
            report += "\n### 築年数別の価格統計（万円）\n"
            report += "| 築年数 | 件数 | 平均価格 | 中央値 | 最小値 | 最大値 |\n"
            report += "|--------|------|----------|--------|--------|--------|\n"
            
            age_stats = df.groupby('AgeGroup')['TradePrice'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index()
            
            for _, row in age_stats.iterrows():
                if not pd.isna(row['AgeGroup']):
                    report += f"| {row['AgeGroup']} | {row['count']} | {row['mean']/10000:,.1f} | {row['median']/10000:,.1f} | {row['min']/10000:,.1f} | {row['max']/10000:,.1f} |\n"
    
    # まとめ（より詳細なものに拡張）
    report += f"\n## まとめ\n"
    
    # 平均価格のコメント
    avg_price = df['TradePrice'].mean() if 'TradePrice' in df.columns else 0
    if avg_price < 30000000:  # 3000万円未満
        price_comment = "比較的手頃な価格帯"
    elif avg_price < 50000000:  # 3000万円〜5000万円
        price_comment = "中価格帯"
    else:  # 5000万円以上
        price_comment = "高価格帯"
    
    # 地区による価格差
    district_comment = ""
    if 'DistrictName' in df.columns and 'TradePrice' in df.columns:
        district_prices = df.groupby('DistrictName')['TradePrice'].mean()
        if len(district_prices) > 3:
            max_district = district_prices.idxmax()
            min_district = district_prices.idxmin()
            price_ratio = district_prices.max() / district_prices.min()
            
            if price_ratio > 2:
                district_comment = f" 地区による価格差が大きく、最も高価な{max_district}と最も安価な{min_district}では約{price_ratio:.1f}倍の差があります。"
            else:
                district_comment = f" 地区による価格差はそれほど大きくなく、最も高価な{max_district}と最も安価な{min_district}でも約{price_ratio:.1f}倍の差にとどまっています。"
    
    # 築年数と価格の関係
    age_comment = ""
    if 'BuildingAge' in df.columns and 'TradePrice' in df.columns:
        corr = df[['BuildingAge', 'TradePrice']].corr().iloc[0, 1]
        
        if corr < -0.5:
            age_comment = f" 築年数と価格には強い負の相関（相関係数:{corr:.2f}）があり、新しい物件ほど価格が高い傾向が顕著です。"
        elif corr < -0.2:
            age_comment = f" 築年数と価格には弱い負の相関（相関係数:{corr:.2f}）があり、比較的新しい物件の方が価格が高い傾向にあります。"
        else:
            age_comment = f" この地域では築年数と価格の相関（相関係数:{corr:.2f}）が弱く、築年数以外の要因が価格に大きく影響していると考えられます。"
    
    # 物件タイプの分布
    type_comment = ""
    if 'Type' in df.columns:
        most_common_type = df['Type'].value_counts().index[0]
        type_percent = 100 * df['Type'].value_counts().iloc[0] / len(df)
        
        if type_percent > 70:
            type_comment = f" 物件タイプは{most_common_type}が圧倒的に多く全体の{type_percent:.1f}%を占めています。"
        elif type_percent > 50:
            type_comment = f" 物件タイプは{most_common_type}が最も多く全体の{type_percent:.1f}%を占めています。"
        else:
            second_type = df['Type'].value_counts().index[1]
            second_percent = 100 * df['Type'].value_counts().iloc[1] / len(df)
            type_comment = f" 物件タイプは{most_common_type}が{type_percent:.1f}%、{second_type}が{second_percent:.1f}%と多様な構成になっています。"
    
    # 期間のトレンド情報
    trend_comment = ""
    if 'Period' in df.columns and 'TradePrice' in df.columns and len(df['Period'].unique()) > 1:
        period_trends = df.groupby('Period')['TradePrice'].mean().sort_index()
        if len(period_trends) > 1:
            first_period = period_trends.index[0]
            last_period = period_trends.index[-1]
            price_change = (period_trends.iloc[-1] / period_trends.iloc[0] - 1) * 100
            
            if abs(price_change) < 5:
                trend_comment = f" {first_period}から{last_period}にかけての価格変動は小さく、市場は比較的安定しています。"
            elif price_change > 0:
                trend_comment = f" {first_period}から{last_period}にかけて平均価格は約{price_change:.1f}%上昇しており、上昇トレンドが見られます。"
            else:
                trend_comment = f" {first_period}から{last_period}にかけて平均価格は約{abs(price_change):.1f}%下落しており、下落トレンドが見られます。"
    
    # 総合的なまとめ文
    if 'Municipality' in df.columns:
        municipality = df['Municipality'].iloc[0] if not df['Municipality'].isna().all() else "対象エリア"
        summary = f"{municipality}は{price_comment}の取引が中心です。{type_comment}{district_comment}{age_comment}{trend_comment}"
    else:
        summary = f"対象エリアは{price_comment}の取引が中心です。{type_comment}{district_comment}{age_comment}{trend_comment}"
    
    report += summary
    
    return report