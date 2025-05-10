# data_handler.py
import pandas as pd
import numpy as np
import requests
import streamlit as st

def get_real_estate_data(api_key, year, area, city):
    """国土交通省APIから不動産取引データを取得する関数"""
    # デバッグモードの確認
    debug_mode = st.session_state.get('debug_mode', False)
    
    # 1. エンドポイントURL
    url = 'https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001?'
    
    # 2. リクエストパラメータ
    params = {
        'year': year,
        'area': area,
        'city': city
    }
    
    # 3. ヘッダー設定
    headers = {
        'Ocp-Apim-Subscription-Key': api_key
    }
    
    # デバッグ情報
    if debug_mode:
        st.write(f"リクエスト情報: URL={url}, パラメータ={params}")
    
    try:
        # 4. リクエスト送信（ヘッダー付き）
        response = requests.get(url, headers=headers, params=params)
        
        # レスポンスのステータスコードとテキスト内容を表示
        if debug_mode:
            st.write(f"レスポンスステータス: {response.status_code}")
        
        # 5. レスポンス確認
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                records = data['data']
                df = pd.DataFrame(records)
                if debug_mode:
                    st.write(f"取得件数: {len(df)} 件")
                return df
            else:
                st.error("データが取得できませんでした。")
                # セッションからデータを削除
                if 'data' in st.session_state:
                    del st.session_state['data']
                return None
        else:
            st.error(f"エラー発生: ステータスコード {response.status_code}")
            # セッションからデータを削除
            if 'data' in st.session_state:
                del st.session_state['data']
            if debug_mode:
                st.write(f"レスポンス内容: {response.text}")
            return None
        
    # 6. エラーハンドリング
    except Exception as e:
        st.error(f"エラー発生: {str(e)}")
        if debug_mode:
            import traceback
            st.write(traceback.format_exc())
        return None
    
def get_municipalities(api_key, prefecture_code):
    """都道府県内の市区町村一覧を取得する関数"""
    url = 'https://www.reinfolib.mlit.go.jp/ex-api/external/XIT002?'
    
    # リクエストパラメータ
    params = {
        'area': prefecture_code
    }
    
    # ヘッダー設定
    headers = {
        'Ocp-Apim-Subscription-Key': api_key
    }
    
    try:
        # リクエスト送信
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                municipalities = data['data']
                return municipalities
            else:
                st.error("市区町村データが取得できませんでした。")
                return []
        else:
            st.error(f"エラー発生: ステータスコード {response.status_code}")
            return []
    except Exception as e:
        st.error(f"エラー発生: {str(e)}")
        return []    

def preprocess_data(df):
    """データの前処理を行う関数"""
    # デバッグモードの確認
    debug_mode = st.session_state.get('debug_mode', False)
    
    try:
        if df is None or df.empty:
            st.error("処理対象のデータがありません")
            return None
        
        # 必要なカラムの抽出
        pic_columns = ['PriceCategory', 'Type', 'MunicipalityCode', 'Prefecture',
                     'Municipality', 'DistrictName', 'TradePrice',
                     'FloorPlan', 'Area', 'BuildingYear', 'Structure', 'CityPlanning',
                     'Period', 'Renovation', 'Remarks']
        
        # 存在するカラムのみ抽出
        available_columns = [col for col in pic_columns if col in df.columns]
        if debug_mode:
            st.write(f"使用可能なカラム: {available_columns}")
        df_pic = df[available_columns]
        
        # BuildingYearの前処理（「年」を取り除く）
        if 'BuildingYear' in df_pic.columns:
            if debug_mode:
                st.write("BuildingYear変換前の例：", df_pic['BuildingYear'].head().tolist())
            # 年の文字を削除し数字のみを抽出
            df_pic['BuildingYear'] = df_pic['BuildingYear'].astype(str).str.replace('年', '').str.replace(r'[^\d.]', '', regex=True)
            if debug_mode:
                st.write("文字列処理後：", df_pic['BuildingYear'].head().tolist())
        
        # 数値型への変換
        numeric_columns = ['TradePrice', 'Area', 'BuildingYear']
        for col in numeric_columns:
            if col in df_pic.columns:
                df_pic.loc[:, col] = pd.to_numeric(df_pic[col], errors='coerce')
                if debug_mode and col == 'BuildingYear':
                    st.write("数値変換後：", df_pic['BuildingYear'].head().tolist())
                    st.write("データ型：", df_pic['BuildingYear'].dtype)
        
        # 築年数の計算
        if 'BuildingYear' in df_pic.columns:
            import datetime
            current_year = datetime.datetime.now().year
            df_pic['BuildingAge'] = current_year - df_pic['BuildingYear']
            if debug_mode:
                st.write("築年数計算後：", df_pic['BuildingAge'].head().tolist())
        
        # 平米あたりの価格計算
        if 'TradePrice' in df_pic.columns and 'Area' in df_pic.columns:
            mask = (df_pic['Area'] > 0) & df_pic['TradePrice'].notna()
            df_pic.loc[mask, 'PricePerSquareMeter'] = df_pic.loc[mask, 'TradePrice'] / df_pic.loc[mask, 'Area']
        
        # ===== 外れ値の検出と処理 =====
        if debug_mode:
            st.write("外れ値処理を開始します...")
            
            # 外れ値処理前のレコード数
            st.write(f"外れ値処理前のレコード数: {len(df_pic)}")
            
            if 'TradePrice' in df_pic.columns:
                st.write("価格の統計情報（処理前）:")
                st.write(df_pic['TradePrice'].describe())
        
        # 外れ値処理の設定（UI経由でカスタマイズ可能にする）
        use_outlier_removal = st.session_state.get('use_outlier_removal', True)
        outlier_method = st.session_state.get('outlier_method', 'iqr')  # 'iqr', 'zscore', 'dbscan'
        outlier_threshold = st.session_state.get('outlier_threshold', 1.5)  # IQR倍数またはZ-score
        
        # 外れ値処理を実行
        if use_outlier_removal:
            # 外れ値除去前のデータを保存
            df_before_outlier = df_pic.copy()
            
            # 価格の外れ値処理
            if 'TradePrice' in df_pic.columns:
                df_pic = remove_outliers(df_pic, 'TradePrice', method=outlier_method, threshold=outlier_threshold)
            
            # 面積の外れ値処理
            if 'Area' in df_pic.columns:
                df_pic = remove_outliers(df_pic, 'Area', method=outlier_method, threshold=outlier_threshold)
            
            # 平米単価の外れ値処理
            if 'PricePerSquareMeter' in df_pic.columns:
                df_pic = remove_outliers(df_pic, 'PricePerSquareMeter', method=outlier_method, threshold=outlier_threshold)
            
            # 築年数の外れ値処理（負の値や極端に古い物件）
            if 'BuildingAge' in df_pic.columns:
                # 負の築年数を除外
                df_pic = df_pic[df_pic['BuildingAge'] >= 0]
                
                # 極端に古い物件を除外（例：100年以上）
                df_pic = df_pic[df_pic['BuildingAge'] <= 100]
            
            # 外れ値処理結果の表示
            if debug_mode:
                st.write(f"外れ値処理後のレコード数: {len(df_pic)}")
                st.write(f"除外された件数: {len(df_before_outlier) - len(df_pic)}")
                
                if 'TradePrice' in df_pic.columns:
                    st.write("価格の統計情報（処理後）:")
                    st.write(df_pic['TradePrice'].describe())

        # ===== 地域特徴量の追加 =====
        use_district_features = st.session_state.get('use_district_features', True)
        if use_district_features:
            df_pic = add_district_features(df_pic)
            if debug_mode:
                st.write("地域特徴量を追加しました")
                
                # 追加された特徴量の確認
                district_cols = [col for col in df_pic.columns if col.startswith('District') or col.startswith('Area_')]
                st.write(f"追加された地域関連の特徴量: {district_cols}")
                
                # 地域クラスタの分布を表示
                if 'DistrictPriceCluster' in df_pic.columns:
                    cluster_counts = df_pic['DistrictPriceCluster'].value_counts().sort_index()
                    st.write("地域クラスタの分布:")
                    st.write(cluster_counts)
        
        # 価格帯の作成
        if 'TradePrice' in df_pic.columns:
            # サンプルデータを確認して適切な閾値を設定
            sample_prices = df_pic['TradePrice'].dropna().head(5)
            if debug_mode:
                st.write(f"サンプル価格: {sample_prices.tolist()}")
            
            # 価格範囲を設定（動的に判断）
            max_price = df_pic['TradePrice'].max()
            if max_price > 1000000:  # 価格が100万円を超える場合、単位は円と判断
                price_bins = [0, 10000000, 30000000, 50000000, 100000000, float('inf')]
            else:  # 価格が小さい場合、単位は万円と判断
                price_bins = [0, 1000, 3000, 5000, 10000, float('inf')]
            
            price_labels = ['〜1,000万円', '1,000〜3,000万円', '3,000〜5,000万円', 
                           '5,000〜1億円', '1億円以上']
            
            if debug_mode:
                st.write(f"価格帯の閾値: {price_bins}")
            df_pic['PriceBin'] = pd.cut(df_pic['TradePrice'], bins=price_bins, labels=price_labels)
        
        # 物件タイプの標準化（表記ゆれの統一）
        if 'Type' in df_pic.columns:
            # 表記ゆれの統一
            type_mapping = {
                'マンション': 'マンション',
                'マンション等': 'マンション', 
                '区分所有建物': 'マンション',
                '中古マンション': 'マンション',
                '一戸建て': '戸建',
                '一戸建': '戸建',
                '中古一戸建て': '戸建',
                '中古一戸建': '戸建',
                '宅地(土地)': '土地',
                '宅地（土地）': '土地',
                '土地': '土地',
                '地上権': '土地'
            }
            
            # マッピングを適用
            df_pic['Type'] = df_pic['Type'].map(lambda x: type_mapping.get(x, x))
        
        if debug_mode:
            st.success("データの前処理が完了しました")
        return df_pic
    
    except Exception as e:
        st.error(f"データの前処理中にエラーが発生しました: {str(e)}")
        if debug_mode:
            import traceback
            st.write(traceback.format_exc())
        # エラーが発生しても、元のデータを返す
        return df

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    指定された列の外れ値を除去する関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        データフレーム
    column : str
        外れ値を検出する列名
    method : str, optional
        外れ値検出方法 ('iqr', 'zscore', 'dbscan')
    threshold : float, optional
        閾値（IQR法の倍数、またはZ-scoreの閾値）
    
    Returns:
    --------
    pandas.DataFrame
        外れ値を除去したデータフレーム
    """
    # 元のデータをコピー
    result_df = df.copy()
    
    # 欠損値を含む行は処理対象外
    valid_data = result_df[result_df[column].notna()]
    
    # データ点が少ない場合は処理しない
    if len(valid_data) < 10:
        return result_df
    
    if method == 'iqr':
        # IQR法（四分位範囲法）による外れ値検出
        Q1 = valid_data[column].quantile(0.25)
        Q3 = valid_data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # 外れ値以外のデータを抽出
        outlier_mask = (result_df[column] >= lower_bound) & (result_df[column] <= upper_bound)
        result_df = result_df[outlier_mask | result_df[column].isna()]
        
    elif method == 'zscore':
        # Z-score法による外れ値検出
        from scipy import stats
        z_scores = stats.zscore(valid_data[column])
        abs_z_scores = np.abs(z_scores)
        
        # Z-scoreがしきい値を超えるデータポイントを特定
        outlier_indices = valid_data.index[abs_z_scores > threshold]
        
        # 外れ値を除外
        result_df = result_df.drop(outlier_indices)
        
    elif method == 'dbscan':
        # DBSCAN法による外れ値検出（クラスタリングベース）
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            # データの標準化
            X = valid_data[[column]].values
            X = StandardScaler().fit_transform(X)
            
            # DBSCANクラスタリング
            db = DBSCAN(eps=threshold, min_samples=5).fit(X)
            labels = db.labels_
            
            # -1のラベルが外れ値
            outlier_indices = valid_data.index[labels == -1]
            
            # 外れ値を除外
            result_df = result_df.drop(outlier_indices)
        except Exception as e:
            # エラーが発生した場合はIQR法にフォールバック
            st.warning(f"DBSCAN法での外れ値検出中にエラーが発生しました: {str(e)}。IQR法を使用します。")
            return remove_outliers(df, column, method='iqr', threshold=threshold)
    
    return result_df    

def add_district_features(df):
    """
    地区名から特徴量を抽出し、地域グループを作成する
    
    Parameters:
    -----------
    df : pandas.DataFrame
        処理対象のデータフレーム
    
    Returns:
    --------
    pandas.DataFrame
        地域特徴量を追加したデータフレーム
    """
    # デバッグモードの確認
    debug_mode = st.session_state.get('debug_mode', False)
    
    if 'DistrictName' not in df.columns:
        if debug_mode:
            st.write("地区名のカラムがないため、地域特徴量を追加できません")
        return df
    
    # 元のデータをコピー
    result_df = df.copy()
    
    try:
        # 地区別の平均価格を計算（重要な特徴量）
        if 'TradePrice' in result_df.columns:
            district_avg_price = result_df.groupby('DistrictName')['TradePrice'].mean()
            # 各データポイントに地区の平均価格を付与
            result_df['DistrictAvgPrice'] = result_df['DistrictName'].map(district_avg_price)
            
            if debug_mode:
                st.write("地区別平均価格を特徴量に追加しました")
        
        # 地区別の物件数をカウント（人気度の指標）
        district_counts = result_df['DistrictName'].value_counts()
        result_df['DistrictCount'] = result_df['DistrictName'].map(district_counts)
        
        if debug_mode:
            st.write("地区別物件数を特徴量に追加しました")
        
        # 地区名から特徴語を抽出 
        # 例: "東京都中央区銀座" から "銀座" を抽出
        def extract_area_name(district):
            if pd.isna(district):
                return None
            
            # 一般的な地区名パターン
            import re
            
            # 「〜丁目」パターンの処理
            chome_match = re.search(r'([^\d]+)(\d+)丁目', district)
            if chome_match:
                base_name = chome_match.group(1)
                return base_name
            
            # 町・区などの接尾辞を持つ場合
            parts = re.split(r'[市区町村]', district)
            if len(parts) > 1:
                return parts[-1]  # 最後の部分を地区名として使用
            
            # その他のケース
            return district
        
        # 地区名の特徴を抽出
        result_df['AreaNameFeature'] = result_df['DistrictName'].apply(extract_area_name)
        
        # 地域グループ化: クラスタリングで地域をグループ化
        # これにより、価格帯が似ている地域をまとめることができる
        if 'TradePrice' in result_df.columns and len(result_df) >= 30:
            try:
                # 地区別の平均価格データを作成
                district_features = result_df.groupby('DistrictName').agg({
                    'TradePrice': ['mean', 'median', 'std'],
                    'DistrictAvgPrice': 'first',
                    'DistrictCount': 'first'
                }).reset_index()
                
                # 標準偏差がNaNの場合（データが1つしかない場合など）は0に置換
                district_features[('TradePrice', 'std')] = district_features[('TradePrice', 'std')].fillna(0)
                
                # カラム名を平坦化
                district_features.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in district_features.columns]
                
                # クラスタリングのための特徴量
                X = district_features[['TradePrice_mean', 'TradePrice_median']].values
                
                # データの標準化
                from sklearn.preprocessing import StandardScaler
                X_scaled = StandardScaler().fit_transform(X)
                
                # クラスタ数を決定 (地区数に応じて可変)
                from sklearn.cluster import KMeans
                
                # 地区の数に応じて適切なクラスタ数を決定
                n_districts = len(district_features)
                if n_districts <= 5:
                    n_clusters = max(2, n_districts - 1)
                elif n_districts <= 20:
                    n_clusters = 5
                else:
                    n_clusters = min(8, n_districts // 5)
                
                # KMeansクラスタリングの実行
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                district_features['PriceCluster'] = kmeans.fit_predict(X_scaled)
                
                # 元のデータフレームにクラスタ情報を結合
                district_cluster_map = dict(zip(district_features['DistrictName'], district_features['PriceCluster']))
                result_df['DistrictPriceCluster'] = result_df['DistrictName'].map(district_cluster_map)
                
                if debug_mode:
                    st.write(f"地域を{n_clusters}個のクラスタにグループ化しました")
                    
                    # クラスタごとの平均価格を表示
                    cluster_info = district_features.groupby('PriceCluster').agg({
                        'TradePrice_mean': 'mean',
                        'DistrictName': 'count'
                    }).reset_index()
                    cluster_info.columns = ['クラスタ', '平均価格', '地区数']
                    st.write("クラスタごとの情報:")
                    st.write(cluster_info)
                
            except Exception as e:
                if debug_mode:
                    st.warning(f"地域クラスタリング中にエラーが発生しました: {str(e)}")
                # エラーが発生しても処理を続行
        
        # 地名の特徴語をOne-Hot Encoding
        if 'AreaNameFeature' in result_df.columns:
            # 出現頻度の低い地名は「その他」にまとめる
            name_counts = result_df['AreaNameFeature'].value_counts()
            min_count = max(3, len(result_df) * 0.02)  # 全体の2%以上、または最低3件
            
            # 頻度の低い地名を「その他」に置換
            result_df['AreaNameGrouped'] = result_df['AreaNameFeature'].apply(
                lambda x: x if pd.notna(x) and name_counts.get(x, 0) >= min_count else 'その他'
            )
            
            # One-Hot Encoding（ダミー変数化）
            area_dummies = pd.get_dummies(result_df['AreaNameGrouped'], prefix='Area')
            
            # 元のデータフレームと結合
            result_df = pd.concat([result_df, area_dummies], axis=1)
            
            if debug_mode:
                st.write(f"地名特徴のダミー変数を{area_dummies.shape[1]}個追加しました")
        
        return result_df
    
    except Exception as e:
        if debug_mode:
            st.error(f"地域特徴量の追加中にエラーが発生しました: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
        
        # エラーが発生しても元のデータフレームを返す
        return df

@st.cache_data
def fetch_and_process_data(api_key, year, area, city):
    """APIからデータを取得し前処理を行う（キャッシュ対応）"""
    # デバッグモードの確認
    debug_mode = st.session_state.get('debug_mode', False)
    
    try:
        raw_data = get_real_estate_data(api_key, year, area, city)
        if raw_data is not None and not raw_data.empty:
            if debug_mode:
                st.write("前処理を開始します...")
            processed_data = preprocess_data(raw_data)
            if processed_data is not None:
                if debug_mode:
                    st.write(f"処理完了: {len(processed_data)}件のデータ")
                return processed_data
            else:
                st.error("データの前処理に失敗しました")
                return raw_data  # 前処理に失敗した場合でも生データを返す
        else:
            st.error("有効なデータが取得できませんでした")
            return None
    except Exception as e:
        st.error(f"データ処理中にエラーが発生しました: {str(e)}")
        if debug_mode:
            import traceback
            st.write(traceback.format_exc())
        return raw_data if 'raw_data' in locals() and raw_data is not None else None