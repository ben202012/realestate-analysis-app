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
    """価格を見やすい形式にフォーマット"""
    if price >= 10000:
        return f"{price/10000:.2f}億円"
    else:
        return f"{price:.0f}万円"

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

def generate_summary_report(df):
    """データの要約レポートを生成"""
    if df is None or df.empty:
        return "データがありません。"
    
    # 基本情報
    report = "# 不動産取引データ 要約レポート\n\n"
    report += f"## 基本情報\n"
    report += f"- データ件数: {len(df)}件\n"
    
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
        report += f"- 平均価格: {df['TradePrice'].mean():,.0f}円\n"
        report += f"- 中央値: {df['TradePrice'].median():,.0f}円\n"
        report += f"- 最小値: {df['TradePrice'].min():,.0f}円\n"
        report += f"- 最大値: {df['TradePrice'].max():,.0f}円\n"
        report += f"- 標準偏差: {df['TradePrice'].std():,.0f}円\n"
    
    # 物件タイプ別の分布
    if 'Type' in df.columns:
        report += f"\n## 物件タイプ別の分布\n"
        type_counts = df['Type'].value_counts()
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
        report += f"\n## 地区別の平均価格（上位5地区）\n"
        district_price = df.groupby('DistrictName')['TradePrice'].agg(['mean', 'count']).reset_index()
        district_price = district_price.sort_values('mean', ascending=False).head(5)
        
        for _, row in district_price.iterrows():
            report += f"- {row['DistrictName']}: {row['mean']:,.0f}円 ({row['count']}件)\n"
    
    # 築年数の統計（ある場合）
    if 'BuildingAge' in df.columns:
        report += f"\n## 築年数の統計\n"
        report += f"- 平均築年数: {df['BuildingAge'].mean():.1f}年\n"
        report += f"- 中央値: {df['BuildingAge'].median():.1f}年\n"
        report += f"- 最小値: {df['BuildingAge'].min():.1f}年\n"
        report += f"- 最大値: {df['BuildingAge'].max():.1f}年\n"
    
    # まとめ
    report += f"\n## まとめ\n"
    # 平均価格のコメント
    avg_price = df['TradePrice'].mean() if 'TradePrice' in df.columns else 0
    if avg_price < 2000000:
        price_comment = "比較的手頃な価格帯"
    elif avg_price < 5000000:
        price_comment = "中価格帯"
    else:
        price_comment = "高価格帯"
    
    report += f"対象エリアは{price_comment}の取引が中心です。"
    
    # 物件タイプのコメント
    if 'Type' in df.columns and not df['Type'].isna().all():
        most_common_type = df['Type'].value_counts().index[0]
        type_percent = 100 * df['Type'].value_counts().iloc[0] / len(df)
        report += f" 物件タイプは{most_common_type}が最も多く全体の{type_percent:.1f}%を占めています。"
    
    return report