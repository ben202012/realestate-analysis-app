# data_handler.py
import pandas as pd
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
        
        # 数値型への変換 - より多くのカラムを対象に
        numeric_columns = ['TradePrice', 'Area', 'BuildingYear', 'BuildingAge']
        for col in numeric_columns:
            if col in df_pic.columns:
                df_pic.loc[:, col] = pd.to_numeric(df_pic[col], errors='coerce')

        # 追加の数値特徴量を作成
        if 'TradePrice' in df_pic.columns and 'Area' in df_pic.columns:
            mask = (df_pic['Area'] > 0) & df_pic['TradePrice'].notna()
            df_pic.loc[mask, 'PricePerSquareMeter'] = df_pic.loc[mask, 'TradePrice'] / df_pic.loc[mask, 'Area']
        
        # 築年数の計算
        if 'BuildingYear' in df_pic.columns:
            import datetime
            current_year = datetime.datetime.now().year
            df_pic['BuildingAge'] = current_year - df_pic['BuildingYear']
        
        # 平米あたりの価格計算
        if 'TradePrice' in df_pic.columns and 'Area' in df_pic.columns:
            mask = (df_pic['Area'] > 0) & df_pic['TradePrice'].notna()
            df_pic.loc[mask, 'PricePerSquareMeter'] = df_pic.loc[mask, 'TradePrice'] / df_pic.loc[mask, 'Area']
        
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
    
    except Exception as e:
        st.error(f"データの前処理中にエラーが発生しました: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        # エラーが発生しても、元のデータを返す
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