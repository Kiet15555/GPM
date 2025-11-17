# TỔNG HỢP TỪ STEP 1 ĐẾN 20 (PHIÊN BẢN ĐẦY ĐỦ - ĐÃ FIX LỖI WEBGL & CỐ ĐỊNH RANDOM SEED)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots 
from scipy.optimize import minimize
import quantstats as qs
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import base64  # SỬA 3: Import để mã hóa ảnh
import streamlit.components.v1 as components  # SỬA 3: Import để dùng HTML

# --- Cấu hình Trang & Các thiết lập ban đầu ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Tối ưu Danh mục VN30 Pro")
# SỬA 1 & 2: Xóa icon và (Bản Full)
# [SỬA 4] CĂN GIỮA TIÊU ĐỀ
st.markdown("<h1 style='text-align: center;'>Ứng dụng Phân tích & Tối ưu hóa Danh mục VN30</h1>", unsafe_allow_html=True)


# --- [SỬA 3] THAY BANNER TĨNH BẰNG SLIDESHOW ---

# Hàm để đọc và mã hóa ảnh sang Base64
def get_image_base64(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

# [SỬA 6] Thêm banner6.jpg vào danh sách
image_paths = ["banner1.jpg", "banner2.jpg", "banner3.jpg", "banner4.jpg", "banner5.jpg", "banner6.jpg"]
base64_images = []
for path in image_paths:
    b64_img = get_image_base64(path)
    if b64_img:
        base64_images.append(b64_img)

if base64_images:
    # Tạo chuỗi HTML cho các ảnh
    html_images = ""
    for b64_img in base64_images:
        # Giả định tất cả là jpeg, có thể cần đổi nếu là png
        html_images += f'<img class="mySlides fade" src="data:image/jpeg;base64,{b64_img}" style="width:100%">'
    
    # Tạo mã HTML/CSS/JS cho slideshow
    html_code = f"""
    <style>
    .slideshow-container {{
      width: 100%;
      position: relative;
      margin: auto;
      /* [SỬA 6] Xóa max-height để ảnh không bị cắt */
      /* max-height: 450px; */ 
      overflow: hidden;
      border-radius: 8px; /* Bo góc */
    }}
    .mySlides {{
      display: none; /* Ẩn tất cả ảnh ban đầu */
      width: 100%;
      /* [SỬA 6] Đặt chiều cao tự động để ảnh không bị cắt */
      height: auto; 
      object-fit: contain; /* [SỬA 6] Đổi từ cover sang contain để ảnh hiển thị toàn bộ */
      vertical-align: middle;
    }}
    /* Hiệu ứng mờ dần */
    .fade {{
      animation-name: fade;
      animation-duration: 3s; /* Giữ nguyên 3s để mềm mại */
    }}
    @keyframes fade {{
      from {{opacity: .4}}
      to {{opacity: 1}}
    }}
    </style>

    <div class="slideshow-container">
      {html_images}
    </div>

    <script>
    let slideIndex = 0;
    showSlides(); // Bắt đầu slideshow

    function showSlides() {{
      let i;
      let slides = document.getElementsByClassName("mySlides");
      if (slides.length === 0) return; // Không có ảnh thì dừng
      
      // Ẩn tất cả ảnh
      for (i = 0; i < slides.length; i++) {{
        slides[i].style.display = "none";
      }}
      
      slideIndex++;
      if (slideIndex > slides.length) {{slideIndex = 1}} // Quay lại ảnh đầu tiên
      
      slides[slideIndex-1].style.display = "block"; // Hiển thị ảnh hiện tại
      
      setTimeout(showSlides, 5000); // Giữ nguyên 5 giây chuyển ảnh
    }}
    </script>
    """
    # [SỬA 6] Để chiều cao của component linh hoạt, có thể bỏ height hoặc dùng 1 giá trị lớn hơn nếu cần.
    # Hoặc để trống nếu muốn nó tự động co giãn hoàn toàn theo nội dung
    components.html(html_code, height=500) # Tăng nhẹ chiều cao mặc định cho HTML component
else:
    st.warning("""
    Không tìm thấy file banner! Vui lòng:
    1. Đổi tên các file ảnh thành: `banner1.jpg`, `banner2.jpg`, `banner3.jpg`, `banner4.jpg`, `banner5.jpg`, `banner6.jpg`
    2. Đặt các file này vào CÙNG THƯ MỤC với tệp `streamlit_app.py`
    """)
# --- KẾT THÚC SỬA BANNER ---


# Set theme mặc định cho Plotly
pio.templates.default = "plotly_dark"
# ... (Phần còn lại của code không thay đổi) ...
# === CÁC HÀM ĐỊNH NGHĨA (GỘP TỪ CÁC FILE) ===

# --- Từ Step2.py (Ô 2) ---
def get_price_history_api(symbol: str, start_date: datetime, end_date: datetime):
    all_data = []
    page = 1
    total_pages = 1
    while page <= total_pages:
        url = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
        params = {"Symbol": symbol, "StartDate": start_date.strftime("%Y-%m-%d"),
                  "EndDate": end_date.strftime("%Y-%m-%d"), "PageIndex": page}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data.get("Success", False): break
            records = data["Data"]["Data"]
            if not records: break
            if page == 1:
                total_count = data["Data"]["TotalCount"]
                total_pages = -(-total_count // len(records))
            all_data.extend(records)
            page += 1
        except Exception as e:
            print(f"Lỗi khi gọi API CafeF cho {symbol}: {e}")
            return None
    if not all_data: return None
    df = pd.DataFrame(all_data)
    df['Ticker'] = symbol.upper()
    numeric_columns = ['GiaDieuChinh', 'GiaDongCua', 'KhoiLuongKhopLenh', 
                      'GiaTriKhopLenh', 'GiaMoCua', 'GiaCaoNhat', 'GiaThapNhat']
    for col in numeric_columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('Ngay', ascending=True).reset_index(drop=True)
    df['GiaDongCua'].replace(0, np.nan, inplace=True); df['GiaDongCua'] = df['GiaDongCua'].ffill().bfill()
    df.loc[df['GiaDongCua'] == 0, 'GiaDieuChinh'] = df['GiaDieuChinh']
    df.loc[df['GiaDongCua'] == 0, 'GiaDongCua'] = 1
    df['adjustment_ratio'] = df['GiaDieuChinh'] / df['GiaDongCua']
    df['open_adj'] = df['GiaMoCua'] * df['adjustment_ratio']
    df['high_adj'] = df['GiaCaoNhat'] * df['adjustment_ratio']
    df['low_adj'] = df['GiaThapNhat'] * df['adjustment_ratio']
    df = df.rename(columns={'Ngay': 'time', 'open_adj': 'open', 'high_adj': 'high',
                            'low_adj': 'low', 'GiaDieuChinh': 'close', 
                            'KhoiLuongKhopLenh': 'volume', 'Ticker': 'ticker'})
    df['time'] = pd.to_datetime(df['time'], format="%d/%m/%Y")
    return df[['time', 'open', 'high', 'low', 'close', 'volume', 'ticker']].sort_values('time').reset_index(drop=True)

# --- Từ Step3.py (Ô 3) ---
def get_stock_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    st.info(f"Bắt đầu lấy dữ liệu cho {len(tickers)} mã (Đa luồng - Thân thiện)...")
    all_data = []
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    def fetch_one_ticker_isolated(ticker):
        time.sleep(0.2) 
        df_ticker = get_price_history_api(ticker, start_dt, end_dt) 
        if df_ticker is not None and not df_ticker.empty:
            return ticker, df_ticker
        else:
            return ticker, None

    progress_text = st.empty()
    progress_bar = st.progress(0)
    total_tickers = len(tickers)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_one_ticker_isolated, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker, df_ticker = future.result()
            completed_count += 1
            progress_bar.progress(completed_count / total_tickers)
            progress_text.text(f"Đang xử lý mã {completed_count}/{total_tickers}: {ticker}...")
            
            if df_ticker is not None:
                all_data.append(df_ticker)
            else:
                st.warning(f"(!) Không tìm thấy dữ liệu cho mã: {ticker}")
    
    progress_text.empty()
    progress_bar.empty()

    if not all_data:
        st.error("(!) Không lấy được bất kỳ dữ liệu nào.")
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(by=['ticker', 'time']).reset_index(drop=True)
    st.success("✅ Lấy dữ liệu (Đa luồng) thành công!")
    return final_df

# --- Hàm tổng hợp từ Step5.py (Ô 4) ---
def load_data(tickers_list, start_time_str, end_time_str, force_refresh):
    CACHE_FILE = 'vn30_data_cache.parquet'
    
    if os.path.exists(CACHE_FILE) and not force_refresh:
        st.info(f"--- Đang tải dữ liệu từ Cache ({CACHE_FILE}) ---")
        try:
            raw_data = pd.read_parquet(CACHE_FILE)
            st.success("✅ Tải từ Cache thành công!")
            return raw_data
        except Exception as e:
            st.warning(f"Lỗi đọc file cache: {e}. Sẽ tải lại từ API.")

    if force_refresh:
        st.warning("--- Bắt buộc làm mới (Force Refresh) ---")
    else:
        st.info("--- Lần chạy đầu tiên (Cache not found) ---")

    raw_data = get_stock_data(tickers_list, start_time_str, end_time_str)

    if not raw_data.empty:
        try:
            st.info(f"--- Đang lưu vào Cache ({CACHE_FILE}) ---")
            raw_data.to_parquet(CACHE_FILE)
            st.success("✅ Dữ liệu đã tải và lưu vào Cache.")
        except Exception as e:
            st.error(f"LỖI khi lưu Cache: {e}. Vui lòng cài đặt 'pip install pyarrow'")
    
    return raw_data

# --- Từ Step7.py (Ô 6) ---
def calculate_stats(returns_df: pd.DataFrame, he_so_scale: int) -> tuple:
    expected_returns = returns_df.mean() * he_so_scale
    cov_matrix = returns_df.cov() * he_so_scale
    return expected_returns, cov_matrix

# --- Từ Step11.py (Ô 10) ---
def get_portfolio_stats(weights: np.array, 
                        expected_returns: pd.Series, 
                        cov_matrix: pd.DataFrame, 
                        risk_free_rate: float) -> tuple:
    port_return = np.sum(weights * expected_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    port_sharpe = (port_return - risk_free_rate) / (port_risk + 1e-9) 
    return (port_return, port_risk, port_sharpe)

def minimize_negative_sharpe(weights: np.array, 
                             expected_returns: pd.Series, 
                             cov_matrix: pd.DataFrame,
                             risk_free_rate: float) -> float:
    return -get_portfolio_stats(weights, expected_returns, cov_matrix, risk_free_rate)[2]

def minimize_portfolio_risk(weights: np.array, 
                            expected_returns: pd.Series, 
                            cov_matrix: pd.DataFrame,
                            risk_free_rate: float) -> float:
    return get_portfolio_stats(weights, expected_returns, cov_matrix, risk_free_rate)[1]

# --- [MỚI] Hàm tính đường biên hiệu quả lý thuyết ---
def calculate_theoretical_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_points=100):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]

    constraints_min_vol = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    opt_min_vol = minimize(minimize_portfolio_risk, init_guess, args=args,
                           method='SLSQP', bounds=bounds, constraints=constraints_min_vol)
    
    min_ret_global = np.sum(mean_returns * opt_min_vol.x)
    max_ret_global = mean_returns.max() 

    target_returns = np.linspace(min_ret_global, max_ret_global, num_points)
    efficient_risks = []
    real_returns = []

    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target}
        )
        opt = minimize(minimize_portfolio_risk, init_guess, args=args,
                       method='SLSQP', bounds=bounds, constraints=constraints)
        if opt.success:
            efficient_risks.append(opt.fun)
            real_returns.append(target)
            
    return pd.DataFrame({'Risk': efficient_risks, 'Return': real_returns})

# --- Từ Step10.py (Ô 8) ---
def run_monte_carlo_sim(n_sims: int, 
                        expected_returns: pd.Series, 
                        cov_matrix: pd.DataFrame, 
                        risk_free_rate: float) -> pd.DataFrame:
    # [FIX] Cố định Seed để kết quả không bị nhảy lung tung mỗi lần chạy
    np.random.seed(42) 
    
    num_assets = len(expected_returns)
    results = np.zeros((3, n_sims))
    weights_record = []
    
    for i in range(n_sims):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        port_return, port_risk, port_sharpe = get_portfolio_stats(
            weights, expected_returns, cov_matrix, risk_free_rate
        )
        
        results[0, i] = port_return
        results[1, i] = port_risk
        results[2, i] = port_sharpe

    results_df = pd.DataFrame(results.T, columns=['Return', 'Risk', 'Sharpe'])
    weights_df = pd.DataFrame(weights_record, columns=expected_returns.index)
    sim_data_df = pd.concat([results_df, weights_df], axis=1)
    
    return sim_data_df

# --- Từ Step16.py (Ô 13) ---
def run_simple_backtest(daily_returns_df: pd.DataFrame, 
                        portfolio_weights: np.array) -> tuple:
    port_returns_daily = daily_returns_df.dot(portfolio_weights)
    port_returns_daily = pd.Series(port_returns_daily, index=daily_returns_df.index)
    cumulative_returns = (1 + port_returns_daily).cumprod()
    return port_returns_daily, cumulative_returns

# --- [MỚI] Hàm phân tích Rebalancing ---
def analyze_rebalancing(returns_df, target_weights, rebalance_freq='Q', transaction_cost=0.001):
    """
    Phân tích chiến lược rebalancing
    rebalance_freq: 'M' (Monthly), 'Q' (Quarterly), 'Y' (Yearly)
    transaction_cost: Chi phí giao dịch (%)
    """
    # Xác định tần suất rebalance (số ngày)
    if rebalance_freq == 'M':
        rebalance_days = 21  # ~1 tháng
    elif rebalance_freq == 'Q':
        rebalance_days = 63  # ~3 tháng
    else:  # 'Y'
        rebalance_days = 252  # ~1 năm
    
    dates = returns_df.index
    n_periods = len(dates)
    
    # Khởi tạo giá trị danh mục
    rebal_value = 1.0
    bh_value = 1.0
    
    rebal_values = [rebal_value]
    bh_values = [bh_value]
    rebalance_dates = []
    total_costs = 0
    
    # Giá trị tuyệt đối của từng asset (không phải %)
    rebal_assets = target_weights.copy() * rebal_value
    bh_assets = target_weights.copy() * bh_value
    
    days_since_rebalance = 0
    
    for i in range(1, n_periods):
        daily_returns = returns_df.iloc[i].values
        days_since_rebalance += 1
        
        # === REBALANCED PORTFOLIO ===
        # Cập nhật giá trị từng asset theo returns
        rebal_assets = rebal_assets * (1 + daily_returns)
        rebal_value = rebal_assets.sum()
        
        # Kiểm tra rebalance
        if days_since_rebalance >= rebalance_days:
            # Tính tỷ trọng hiện tại
            current_weights = rebal_assets / rebal_value
            
            # Chi phí giao dịch
            turnover = np.abs(current_weights - target_weights).sum()
            cost = turnover * transaction_cost
            total_costs += cost
            
            # Trừ chi phí
            rebal_value = rebal_value * (1 - cost)
            
            # Rebalance về target weights
            rebal_assets = target_weights * rebal_value
            
            rebalance_dates.append(dates[i])
            days_since_rebalance = 0
        
        rebal_values.append(rebal_value)
        
        # === BUY & HOLD ===
        bh_assets = bh_assets * (1 + daily_returns)
        bh_value = bh_assets.sum()
        bh_values.append(bh_value)
    
    return {
        'rebalanced_value': pd.Series(rebal_values, index=dates),
        'buy_hold_value': pd.Series(bh_values, index=dates),
        'rebalance_dates': rebalance_dates,
        'total_costs': total_costs,
        'num_rebalances': len(rebalance_dates)
    }

# --- [MỚI] Hàm export Excel ---
def export_to_excel(weights_df, summary_table, returns_df, price_pivot):
    """Xuất kết quả ra file Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Tỷ trọng tối ưu
        weights_df.to_excel(writer, sheet_name='Tỷ trọng tối ưu')
        
        # Sheet 2: Bảng tổng kết
        summary_table.to_excel(writer, sheet_name='Tổng kết hiệu suất')
        
        # Sheet 3: Returns
        returns_df.to_excel(writer, sheet_name='Tỷ suất sinh lời')
        
        # Sheet 4: Prices
        price_pivot.to_excel(writer, sheet_name='Giá đóng cửa')
        
        # Sheet 5: Hướng dẫn
        instructions = pd.DataFrame({
            'Sheet': ['Tỷ trọng tối ưu', 'Tổng kết hiệu suất', 'Tỷ suất sinh lời', 'Giá đóng cửa'],
            'Mô tả': [
                'Phân bổ tỷ trọng % cho 3 chiến lược đầu tư',
                'Các chỉ số đánh giá hiệu suất (CAGR, Sharpe, Drawdown...)',
                'Tỷ suất sinh lời hàng ngày của từng cổ phiếu',
                'Giá đóng cửa điều chỉnh của từng cổ phiếu'
            ]
        })
        instructions.to_excel(writer, sheet_name='Hướng dẫn', index=False)
    
    return output.getvalue()

# --- [MỚI] Phân loại Sector ---
SECTOR_MAPPING = {
    # Banking
    'ACB': 'Ngân hàng', 'BID': 'Ngân hàng', 'CTG': 'Ngân hàng', 'HDB': 'Ngân hàng',
    'LPB': 'Ngân hàng', 'MBB': 'Ngân hàng', 'SHB': 'Ngân hàng', 'SSB': 'Ngân hàng',
    'STB': 'Ngân hàng', 'TCB': 'Ngân hàng', 'TPB': 'Ngân hàng', 'VCB': 'Ngân hàng',
    'VIB': 'Ngân hàng', 'VPB': 'Ngân hàng',
    
    # Real Estate
    'VHM': 'Bất động sản', 'VIC': 'Bất động sản', 'VRE': 'Bất động sản',
    
    # Oil & Gas
    'GAS': 'Dầu khí', 'PLX': 'Dầu khí', 'GVR': 'Dầu khí',
    
    # Manufacturing
    'HPG': 'Sản xuất', 'GVR': 'Sản xuất',
    
    # Consumer
    'MSN': 'Tiêu dùng', 'MWG': 'Tiêu dùng', 'SAB': 'Tiêu dùng', 'VNM': 'Tiêu dùng',
    
    # Technology
    'FPT': 'Công nghệ',
    
    # Insurance
    'BVH': 'Bảo hiểm',
    
    # Aviation
    'VJC': 'Hàng không',
    
    # Securities
    'SSI': 'Chứng khoán',
    
    # Mining
    'BCM': 'Khai khoáng'
}

def get_sector_allocation(weights_df):
    """Tính phân bổ theo ngành"""
    sector_data = {}
    for portfolio in weights_df.columns:
        sector_weights = {}
        for ticker, weight in weights_df[portfolio].items():
            if weight > 0.001:
                sector = SECTOR_MAPPING.get(ticker, 'Khác')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
        sector_data[portfolio] = sector_weights
    return pd.DataFrame(sector_data).fillna(0)

# --- [MỚI] Machine Learning: Dự đoán Return ---
def ml_predict_returns(returns_df, price_pivot):
    """Sử dụng Random Forest để dự đoán returns"""
    predictions = {}
    feature_importance = {}
    
    for ticker in returns_df.columns:  # Train cho TẤT CẢ cổ phiếu
        try:
            # Tạo features
            df = pd.DataFrame()
            df['return'] = returns_df[ticker]
            df['return_lag1'] = df['return'].shift(1)
            df['return_lag2'] = df['return'].shift(2)
            df['return_lag3'] = df['return'].shift(3)
            df['return_ma5'] = df['return'].rolling(5).mean()
            df['return_ma20'] = df['return'].rolling(20).mean()
            df['volatility_20'] = df['return'].rolling(20).std()
            df = df.dropna()
            
            if len(df) < 100:
                continue
            
            # Train/Test split
            train_size = int(len(df) * 0.8)
            X_train = df.iloc[:train_size, 1:]
            y_train = df.iloc[:train_size, 0]
            X_test = df.iloc[train_size:, 1:]
            y_test = df.iloc[train_size:, 0]
            
            # Train model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            predictions[ticker] = {
                'actual': y_test.values,
                'predicted': y_pred,
                'score': model.score(X_test_scaled, y_test)
            }
            
            feature_importance[ticker] = pd.Series(
                model.feature_importances_,
                index=X_train.columns
            )
        except:
            continue
    
    return predictions, feature_importance

# --- [MỚI] Clustering cổ phiếu ---
def cluster_stocks(returns_df, n_clusters=3):
    """Phân nhóm cổ phiếu theo đặc tính"""
    # Tính features cho mỗi cổ phiếu
    features = pd.DataFrame({
        'mean_return': returns_df.mean(),
        'volatility': returns_df.std(),
        'sharpe': returns_df.mean() / returns_df.std(),
        'skewness': returns_df.skew(),
        'kurtosis': returns_df.kurtosis()
    })
    
    # Chuẩn hóa
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    features['Cluster'] = clusters
    features['Ticker'] = features.index
    
    return features


# === GIAO DIỆN STREAMLIT ===

# --- Sidebar (Thanh bên) ---
st.sidebar.header("Cấu hình Phân tích")

start_date_input = st.sidebar.date_input(
    'Từ ngày', value=datetime(2018, 1, 1)
)
end_date_input = st.sidebar.date_input(
    'Đến ngày', value=datetime.now()
)

holding_period_tuple = st.sidebar.selectbox(
    'Thời hạn (Scale):',
    options=[('1 năm', 252), ('6 tháng', 126), ('2 năm', 504), ('3 tháng', 63)],
    format_func=lambda x: x[0]
)
HE_SO_SCALE = holding_period_tuple[1]

risk_free_rate_pct = st.sidebar.number_input(
    'LS Phi rủi ro (%):', value=4.0, step=0.1
)
RISK_FREE_RATE = risk_free_rate_pct / 100.0

N_SIMULATIONS = st.sidebar.number_input(
    'Số lần Mô phỏng Monte Carlo:',
    min_value=1000, max_value=50000, value=5000, step=1000
)

# THÊM: Giới hạn tỷ trọng để đa dạng hóa
st.sidebar.subheader("⚖️ Đa dạng hóa Danh mục")
max_weight_pct = st.sidebar.slider(
    'Tỷ trọng tối đa mỗi cổ phiếu (%):', 
    min_value=5, max_value=100, value=30, step=5,
    help="Giới hạn tỷ trọng để tránh tập trung rủi ro vào 1 cổ phiếu"
)
MAX_WEIGHT = max_weight_pct / 100.0

min_stocks = st.sidebar.number_input(
    'Số cổ phiếu tối thiểu trong danh mục:',
    min_value=3, max_value=30, value=5, step=1,
    help="Đảm bảo danh mục có ít nhất X cổ phiếu để đa dạng"
)

force_refresh_checkbox = st.sidebar.checkbox(
    'Làm mới Dữ liệu (Bỏ qua Cache & gọi lại API)'
)

run_button = st.sidebar.button("BẮT ĐẦU PHÂN TÍCH", type="primary", use_container_width=True)

tickers_list = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

# === QUY TRÌNH CHÍNH ===

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if run_button:
    st.session_state.analysis_done = False 
    
    # 1. Tải Dữ liệu
    with st.spinner("⏳ (1/7) Đang tải dữ liệu thô..."):
        start_time_str = start_date_input.strftime('%Y-%m-%d')
        end_time_str = end_date_input.strftime('%Y-%m-%d')
        
        raw_data = load_data(tickers_list, start_time_str, end_time_str, force_refresh_checkbox)
        
        if raw_data.empty:
            st.error("Không tải được dữ liệu. Vui lòng thử lại.")
            st.stop()
        
        st.session_state.raw_data = raw_data
        st.session_state.start_time_str = start_time_str
    
    # 2. Tính Tỷ suất sinh lời
    with st.spinner("⏳ (2/7) Đang tính Tỷ suất sinh lời..."):
        raw_data.drop_duplicates(subset=['time', 'ticker'], keep='last', inplace=True)
        price_pivot = raw_data.pivot(index='time', columns='ticker', values='close')
        returns_df_raw = price_pivot.pct_change()
        returns_df = returns_df_raw.iloc[1:].dropna(how='all') 
        
        st.session_state.returns_df = returns_df
        st.session_state.price_pivot = price_pivot
    
    # 3. Tính Stats
    with st.spinner("⏳ (3/7) Đang tính Lợi nhuận Kỳ vọng & Hiệp phương sai..."):
        expected_returns, cov_matrix = calculate_stats(returns_df, HE_SO_SCALE)
        
        valid_assets = expected_returns.dropna().index
        expected_returns = expected_returns[valid_assets]
        cov_matrix = cov_matrix.loc[valid_assets, valid_assets]
        
        st.session_state.expected_returns = expected_returns
        st.session_state.cov_matrix = cov_matrix
        st.session_state.returns_df = returns_df[valid_assets] 
        st.session_state.price_pivot = price_pivot[valid_assets]
    
    # 4. Chạy Monte Carlo VÀ Efficient Frontier
    with st.spinner(f"⏳ (4/7) Đang chạy Monte Carlo & Dựng đường biên hiệu quả..."):
        sim_data_df = run_monte_carlo_sim(
            N_SIMULATIONS, 
            st.session_state.expected_returns, 
            st.session_state.cov_matrix, 
            RISK_FREE_RATE
        )
        st.session_state.sim_data_df = sim_data_df

        eff_frontier_df = calculate_theoretical_efficient_frontier(
            st.session_state.expected_returns, 
            st.session_state.cov_matrix, 
            RISK_FREE_RATE
        )
        st.session_state.eff_frontier_df = eff_frontier_df

    # 5. Chạy Tối ưu hóa (CÓ GIỚI HẠN ĐA DẠNG HÓA)
    with st.spinner("⏳ (5/7) Đang chạy Tối ưu hóa với giới hạn đa dạng..."):
        num_assets = len(st.session_state.expected_returns)
        args = (st.session_state.expected_returns, st.session_state.cov_matrix, RISK_FREE_RATE)
        
        # Ràng buộc: Tổng = 1, Tỷ trọng <= MAX_WEIGHT, Số cổ phiếu >= min_stocks
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Tổng = 100%
        ]
        bounds = tuple((0.0, MAX_WEIGHT) for _ in range(num_assets))  # Giới hạn từng cổ phiếu

        # 1. Min Risk với đa dạng hóa
        min_vol_guess_weights = sim_data_df.loc[sim_data_df['Risk'].idxmin()].values[3:3+num_assets]
        min_vol_guess_weights = np.clip(min_vol_guess_weights, 0, MAX_WEIGHT)
        min_vol_guess_weights /= min_vol_guess_weights.sum()
        
        opt_min_vol = minimize(minimize_portfolio_risk, min_vol_guess_weights, args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)
        min_vol_weights = opt_min_vol.x
        
        # Đảm bảo có đủ số cổ phiếu
        if np.sum(min_vol_weights > 0.001) < min_stocks:
            # Phân bổ đều cho top N cổ phiếu có risk thấp nhất
            top_n_idx = np.argsort(np.diag(st.session_state.cov_matrix))[:min_stocks]
            min_vol_weights = np.zeros(num_assets)
            min_vol_weights[top_n_idx] = 1.0 / min_stocks

        # 2. Max Sharpe với đa dạng hóa
        max_sharpe_guess_weights = sim_data_df.loc[sim_data_df['Sharpe'].idxmax()].values[3:3+num_assets]
        max_sharpe_guess_weights = np.clip(max_sharpe_guess_weights, 0, MAX_WEIGHT)
        max_sharpe_guess_weights /= max_sharpe_guess_weights.sum()
        
        opt_max_sharpe = minimize(minimize_negative_sharpe, max_sharpe_guess_weights, args=args,
                                  method='SLSQP', bounds=bounds, constraints=constraints)
        max_sharpe_weights = opt_max_sharpe.x
        
        if np.sum(max_sharpe_weights > 0.001) < min_stocks:
            top_n_idx = np.argsort(-st.session_state.expected_returns)[:min_stocks]
            max_sharpe_weights = np.zeros(num_assets)
            max_sharpe_weights[top_n_idx] = 1.0 / min_stocks
        
        # 3. Max Return với đa dạng hóa (không cho phép 100% vào 1 cổ phiếu)
        top_n_returns_idx = np.argsort(-st.session_state.expected_returns)[:min_stocks]
        max_ret_weights = np.zeros(num_assets)
        
        # Phân bổ theo tỷ lệ lợi nhuận của top stocks
        top_returns = st.session_state.expected_returns.values[top_n_returns_idx]
        top_returns = np.maximum(top_returns, 0)  # Chỉ lấy returns dương
        if top_returns.sum() > 0:
            max_ret_weights[top_n_returns_idx] = top_returns / top_returns.sum()
            max_ret_weights = np.clip(max_ret_weights, 0, MAX_WEIGHT)
            max_ret_weights /= max_ret_weights.sum()
        else:
            # Nếu không có returns dương, phân bổ đều
            max_ret_weights[top_n_returns_idx] = 1.0 / min_stocks

        st.session_state.optimal_weights_df = pd.DataFrame({
            'Bảo thủ (Min Risk)': min_vol_weights,
            'Cân bằng (Max Sharpe)': max_sharpe_weights,
            'Mạo hiểm (Max Return)': max_ret_weights
        }, index=st.session_state.expected_returns.index)
        
        st.session_state.optimal_weights_df.index.name = 'ticker'

        st.session_state.optimal_stats_dict = {
            'min_vol': get_portfolio_stats(min_vol_weights, *args),
            'max_sharpe': get_portfolio_stats(max_sharpe_weights, *args),
            'max_ret': get_portfolio_stats(max_ret_weights, *args)
        }
        
        # THÊM: Tính số cổ phiếu thực tế trong mỗi danh mục
        st.session_state.num_stocks = {
            'Bảo thủ (Min Risk)': np.sum(min_vol_weights > 0.001),
            'Cân bằng (Max Sharpe)': np.sum(max_sharpe_weights > 0.001),
            'Mạo hiểm (Max Return)': np.sum(max_ret_weights > 0.001)
        }

    # 6. Chạy Backtest
    with st.spinner("⏳ (6/7) Đang chạy Backtest..."):
        returns_min_vol, cum_min_vol = run_simple_backtest(st.session_state.returns_df, min_vol_weights)
        returns_max_sharpe, cum_max_sharpe = run_simple_backtest(st.session_state.returns_df, max_sharpe_weights)
        returns_max_ret, cum_max_ret = run_simple_backtest(st.session_state.returns_df, max_ret_weights)
        
        st.session_state.all_cumulative_df = pd.DataFrame({
            'Bảo thủ (Min Risk)': cum_min_vol,
            'Cân bằng (Max Sharpe)': cum_max_sharpe,
            'Mạo hiểm (Max Return)': cum_max_ret
        }).dropna()
        
        st.session_state.all_returns_df = pd.DataFrame({
            'Bảo thủ (Min Risk)': returns_min_vol,
            'Cân bằng (Max Sharpe)': returns_max_sharpe,
            'Mạo hiểm (Max Return)': returns_max_ret
        }).dropna()

    # 7. Tính Metrics
    with st.spinner("⏳ (7/7) Đang tính toán Metrics hiệu suất..."):
        metrics = ['Tổng Lợi nhuận (Cumulative)', 'Lợi nhuận TB Năm (Annualized)', 
                   'Rủi ro Năm (Annualized)', 'Mức sụt giảm Tối đa (Max Drawdown)', 
                   'Chỉ số Sharpe (Historical)']
        summary_table = pd.DataFrame(index=metrics)
        
        for port_name in st.session_state.all_returns_df.columns:
            returns_series = st.session_state.all_returns_df[port_name]
            summary_table.loc['Tổng Lợi nhuận (Cumulative)', port_name] = qs.stats.comp(returns_series)
            summary_table.loc['Lợi nhuận TB Năm (Annualized)', port_name] = qs.stats.cagr(returns_series)
            summary_table.loc['Rủi ro Năm (Annualized)', port_name] = qs.stats.volatility(returns_series)
            summary_table.loc['Mức sụt giảm Tối đa (Max Drawdown)', port_name] = qs.stats.max_drawdown(returns_series)
            summary_table.loc['Chỉ số Sharpe (Historical)', port_name] = qs.stats.sharpe(returns_series, rf=RISK_FREE_RATE)
        
        st.session_state.summary_table = summary_table
    
    st.session_state.analysis_done = True
    st.success("🎉 Phân tích hoàn tất! Xem kết quả bên dưới.")

# === HIỂN THỊ KẾT QUẢ ===

if st.session_state.analysis_done:
    
    # THÊM: Nút Export ở đầu
    st.markdown("---")
    col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
    
    with col_export1:
        # Export Excel
        excel_data = export_to_excel(
            st.session_state.optimal_weights_df,
            st.session_state.summary_table,
            st.session_state.returns_df,
            st.session_state.price_pivot
        )
        st.download_button(
            label="📥 Tải xuống Excel",
            data=excel_data,
            file_name=f"Portfolio_Analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col_export2:
        # Export CSV
        csv_data = st.session_state.optimal_weights_df.to_csv()
        st.download_button(
            label="📥 Tải xuống CSV",
            data=csv_data,
            file_name=f"Portfolio_Weights_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col_export3:
        st.info("💾 Tải xuống kết quả phân tích để lưu trữ hoặc báo cáo")
    
    st.markdown("---")
    
    # THÊM: Thêm tab mới
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dữ liệu & Tương quan", 
        "🎯 Tỷ trọng & Đường biên", 
        "📈 Backtest & Hiệu suất",
        "🛡️ Rủi ro & Đa dạng",
        "🏢 Phân tích Ngành & ML"
    ])

    # Tab 1: Dữ liệu Thô & Tương quan
    with tab1:
        st.header("📊 Dữ liệu Thô & Phân tích Tương quan")
        
        # THÊM: Thống kê mô tả
        st.subheader("Thống kê Mô tả Tỷ suất Sinh lời")
        returns_df = st.session_state.returns_df
        desc_stats = returns_df.describe().T
        desc_stats['skew'] = returns_df.skew()
        desc_stats['kurtosis'] = returns_df.kurtosis()
        desc_stats_styled = desc_stats.style.format({
            'mean': '{:.4f}',
            'std': '{:.4f}',
            'min': '{:.4f}',
            '25%': '{:.4f}',
            '50%': '{:.4f}',
            '75%': '{:.4f}',
            'max': '{:.4f}',
            'skew': '{:.2f}',
            'kurtosis': '{:.2f}'
        })
        st.dataframe(desc_stats_styled, use_container_width=True)
        
        # THÊM: Biểu đồ phân phối lợi nhuận
        st.subheader("Phân phối Lợi nhuận Hàng ngày (Top 10 cổ phiếu)")
        top_10_tickers = st.session_state.expected_returns.nlargest(10).index.tolist()
        fig_dist = go.Figure()
        for ticker in top_10_tickers:
            fig_dist.add_trace(go.Histogram(
                x=returns_df[ticker].dropna(),
                name=ticker,
                opacity=0.6,
                nbinsx=50
            ))
        fig_dist.update_layout(
            title='Phân phối Lợi nhuận Hàng ngày (Top 10 cổ phiếu theo Return kỳ vọng)',
            xaxis_title='Tỷ suất sinh lời',
            yaxis_title='Tần suất',
            barmode='overlay',
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.subheader("Heatmap Ma trận Tương quan")
        correlation_matrix = returns_df.corr()
        labels = correlation_matrix.columns
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values, x=labels, y=labels,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            hoverongaps=False,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8}
        ))
        fig_heatmap.update_layout(
            title='Heatmap Ma trận Tương quan (VN30)', 
            template='plotly_dark',
            height=800, width=900,
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # THÊM: Biểu đồ cặp tương quan cao nhất
        st.subheader("Top 10 cặp cổ phiếu có Tương quan cao nhất")
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append({
                    'Cặp': f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}",
                    'Tương quan': correlation_matrix.iloc[i, j]
                })
        corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Tương quan', ascending=False).head(10)
        fig_corr_pairs = px.bar(
            corr_pairs_df, x='Tương quan', y='Cặp', orientation='h',
            title='Top 10 cặp cổ phiếu có Tương quan cao nhất',
            text='Tương quan'
        )
        fig_corr_pairs.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_corr_pairs.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig_corr_pairs, use_container_width=True)
        
        st.subheader("Dữ liệu Giá Đóng cửa (Pivot - 10 dòng cuối)")
        st.dataframe(st.session_state.price_pivot.tail(10))
        
        st.subheader("Dữ liệu Tỷ suất sinh lời (Hàng ngày - 10 dòng cuối)")
        st.dataframe(st.session_state.returns_df.tail(10))

    # Tab 2: Phân bổ Tỷ trọng & Đường biên
    with tab2:
        st.header("🎯 Phân bổ Tỷ trọng & Đường biên Hiệu quả")
        
        # THÊM: Hiển thị thông tin đa dạng hóa
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🛡️ Tỷ trọng tối đa", f"{MAX_WEIGHT*100:.0f}%")
        with col2:
            st.metric("📊 Số cổ phiếu tối thiểu", f"{min_stocks} cổ phiếu")
        with col3:
            avg_stocks = np.mean(list(st.session_state.num_stocks.values()))
            st.metric("📈 Số cổ phiếu TB trong danh mục", f"{avg_stocks:.1f} cổ phiếu")
        
        st.divider()
        
        # --- PHẦN 1: BẢNG SỐ LIỆU ---
        st.subheader("📋 Phân bổ Tỷ trọng Tối ưu")
        df_weights = st.session_state.optimal_weights_df
        df_weights_display = df_weights[(df_weights > 0.001).any(axis=1)].copy()
        
        # THÊM: Số lượng cổ phiếu trong mỗi danh mục
        num_stocks_row = pd.DataFrame({
            col: [f"{st.session_state.num_stocks[col]} cổ phiếu"] 
            for col in df_weights.columns
        }, index=['Số lượng CP'])
        
        st.info(f"**Nguyên tắc đa dạng hóa**: Mỗi cổ phiếu không vượt quá {MAX_WEIGHT*100:.0f}%, đảm bảo ít nhất {min_stocks} cổ phiếu trong danh mục")
        
        # Hiển thị số lượng cổ phiếu
        st.dataframe(num_stocks_row, use_container_width=True)
        
        df_weights_styled = df_weights_display.style.format("{:.2%}").background_gradient(
            cmap='RdYlGn', axis=0, vmin=0, vmax=MAX_WEIGHT
        )
        st.dataframe(df_weights_styled, use_container_width=True)
        
        st.divider()
        
        # --- PHẦN 2: BIỂU ĐỒ TRÒN TỶ TRỌNG ---
        st.subheader("Biểu đồ Tròn - Cơ cấu Danh mục")
        selected_portfolio = st.selectbox(
            "Chọn danh mục để xem chi tiết:",
            options=df_weights.columns.tolist()
        )
        
        weights_selected = df_weights[selected_portfolio]
        weights_selected = weights_selected[weights_selected > 0.001].sort_values(ascending=False)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=weights_selected.index,
            values=weights_selected.values,
            hole=0.4,
            textposition='auto',
            textinfo='label+percent',
            marker=dict(line=dict(color='#000000', width=2))
        )])
        fig_pie.update_layout(
            title=f'Cơ cấu Danh mục: {selected_portfolio}',
            template='plotly_dark',
            height=600,
            annotations=[dict(text=f'{len(weights_selected)}<br>cổ phiếu', 
                            x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.divider()
        
        # --- PHẦN 3: BIỂU ĐỒ TỶ TRỌNG SO SÁNH ---
        st.subheader("Biểu đồ Tỷ trọng - So sánh 3 Danh mục")
        df_plot = df_weights[(df_weights > 0.001).any(axis=1)].copy()
        
        df_plot_long = df_plot.reset_index().melt(
            id_vars='ticker', 
            var_name='Danh mục', 
            value_name='Tỷ trọng'
        )
        df_plot_long.rename(columns={'ticker': 'Mã CP'}, inplace=True)
        
        fig_bars = px.bar(
            df_plot_long, x='Danh mục', y='Tỷ trọng', color='Mã CP',
            text_auto='.1%',
            title='Phân bổ Tỷ trọng Tối ưu theo 3 Khẩu vị Rủi ro'
        )
        fig_bars.update_layout(template='plotly_dark', yaxis_tickformat='.0%', height=600)
        fig_bars.add_hline(y=MAX_WEIGHT, line_dash="dash", line_color="red", 
                          annotation_text=f"Giới hạn {MAX_WEIGHT*100:.0f}%")
        st.plotly_chart(fig_bars, use_container_width=True)
            
        st.divider()

        # --- PHẦN 4: BIỂU ĐỒ ĐƯỜNG BIÊN HIỆU QUẢ ---
        st.subheader("Đường biên Hiệu quả Toàn diện (có CAL)")
        
        # FIX: THÊM render_mode='svg' ĐỂ SỬA LỖI WEBGL
        sim_data_df = st.session_state.sim_data_df
        fig = px.scatter(
            sim_data_df, x='Risk', y='Return', color='Sharpe',
            color_continuous_scale='Viridis',
            hover_data={col: ':.2%' for col in sim_data_df.columns if col not in ['Risk', 'Return', 'Sharpe']} | {'Risk': ':.2%','Return': ':.2%','Sharpe': ':.2f'},
            title=f'Đường biên Hiệu quả - {N_SIMULATIONS} danh mục (Rf={RISK_FREE_RATE:.1%}, Max Weight={MAX_WEIGHT:.0%})',
            render_mode='svg' # <--- ĐÂY LÀ FIX QUAN TRỌNG
        )
        
        # Vẽ đường lý thuyết
        eff_df = st.session_state.eff_frontier_df
        fig.add_trace(go.Scatter(
            x=eff_df['Risk'], y=eff_df['Return'], mode='lines', 
            line=dict(color='white', width=3, dash='dash'),
            name='Đường biên Hiệu quả (Lý thuyết)'
        ))
        
        # Vẽ các điểm tối ưu
        stats_dict = st.session_state.optimal_stats_dict
        stats_min_vol = stats_dict['min_vol']
        stats_max_sharpe = stats_dict['max_sharpe']
        stats_max_ret = stats_dict['max_ret']
        
        fig.add_trace(go.Scatter(x=[stats_min_vol[1]], y=[stats_min_vol[0]], mode='markers', 
                                marker=dict(color='white', size=20, symbol='star', 
                                          line=dict(color='black', width=2)), 
                                name='Bảo thủ (Min Risk)',
                                hovertemplate='<b>Bảo thủ</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}'))
        fig.add_trace(go.Scatter(x=[stats_max_sharpe[1]], y=[stats_max_sharpe[0]], mode='markers', 
                                marker=dict(color='cyan', size=20, symbol='star', 
                                          line=dict(color='black', width=2)), 
                                name='Cân bằng (Max Sharpe)',
                                hovertemplate='<b>Cân bằng</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}'))
        fig.add_trace(go.Scatter(x=[stats_max_ret[1]], y=[stats_max_ret[0]], mode='markers', 
                                marker=dict(color='red', size=20, symbol='star', 
                                          line=dict(color='black', width=2)), 
                                name='Mạo hiểm (Max Return)',
                                hovertemplate='<b>Mạo hiểm</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}'))
        
        # Vẽ Đường CAL
        sharpe_risk = stats_max_sharpe[1]
        sharpe_return = stats_max_sharpe[0]
        x_cal = [0, sharpe_risk * 1.5] 
        y_cal = [RISK_FREE_RATE, (sharpe_return - RISK_FREE_RATE) / (sharpe_risk + 1e-9) * (sharpe_risk * 1.5) + RISK_FREE_RATE]
        fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', 
                                line=dict(color='lime', width=3, dash='dash'), 
                                name='Đường Phân bổ Vốn (CAL)'))

        fig.update_layout(
            height=800,
            xaxis_tickformat='.1%', yaxis_tickformat='.1%',
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(b=120)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # THÊM: Bảng so sánh 3 danh mục
        st.subheader("So sánh Chỉ số của 3 Danh mục Tối ưu")
        comparison_df = pd.DataFrame({
            'Bảo thủ (Min Risk)': [
                f"{stats_min_vol[0]:.2%}", 
                f"{stats_min_vol[1]:.2%}", 
                f"{stats_min_vol[2]:.2f}",
                f"{st.session_state.num_stocks['Bảo thủ (Min Risk)']} cổ phiếu",
                f"{df_weights['Bảo thủ (Min Risk)'].max():.2%}"
            ],
            'Cân bằng (Max Sharpe)': [
                f"{stats_max_sharpe[0]:.2%}", 
                f"{stats_max_sharpe[1]:.2%}", 
                f"{stats_max_sharpe[2]:.2f}",
                f"{st.session_state.num_stocks['Cân bằng (Max Sharpe)']} cổ phiếu",
                f"{df_weights['Cân bằng (Max Sharpe)'].max():.2%}"
            ],
            'Mạo hiểm (Max Return)': [
                f"{stats_max_ret[0]:.2%}", 
                f"{stats_max_ret[1]:.2%}", 
                f"{stats_max_ret[2]:.2f}",
                f"{st.session_state.num_stocks['Mạo hiểm (Max Return)']} cổ phiếu",
                f"{df_weights['Mạo hiểm (Max Return)'].max():.2%}"
            ]
        }, index=['Lợi nhuận Kỳ vọng', 'Rủi ro (Độ lệch chuẩn)', 'Chỉ số Sharpe', 'Số lượng CP', 'Tỷ trọng CP lớn nhất'])
        st.dataframe(comparison_df, use_container_width=True)

    # Tab 3: Kết quả Backtest
    with tab3:
        st.header("📈 Kết quả Backtest & Hiệu suất")
        
        st.subheader("Bảng Tổng kết Chỉ số Hiệu suất")
        summary_table = st.session_state.summary_table
        percent_rows = summary_table.index.difference(['Chỉ số Sharpe (Historical)'])
        number_row = pd.Index(['Chỉ số Sharpe (Historical)'])
        styler = summary_table.style
        styler.format('{:,.2%}', subset=(percent_rows, slice(None)))
        styler.format('{:,.2f}', subset=(number_row, slice(None)))
        
        # THÊM: Highlight giá trị tốt nhất
        styler.highlight_max(axis=1, color='lightgreen', subset=pd.IndexSlice[['Tổng Lợi nhuận (Cumulative)', 'Lợi nhuận TB Năm (Annualized)', 'Chỉ số Sharpe (Historical)'], :])
        styler.highlight_min(axis=1, color='lightgreen', subset=pd.IndexSlice[['Rủi ro Năm (Annualized)', 'Mức sụt giảm Tối đa (Max Drawdown)'], :])
        
        st.dataframe(styler, use_container_width=True)
        
        st.divider()
        
        # THÊM: Các metrics nổi bật
        col1, col2, col3 = st.columns(3)
        with col1:
            best_return = summary_table.loc['Lợi nhuận TB Năm (Annualized)'].idxmax()
            best_return_val = summary_table.loc['Lợi nhuận TB Năm (Annualized)', best_return]
            st.metric("🏆 Danh mục có Return cao nhất", best_return, f"{best_return_val:.2%}")
        
        with col2:
            best_sharpe = summary_table.loc['Chỉ số Sharpe (Historical)'].idxmax()
            best_sharpe_val = summary_table.loc['Chỉ số Sharpe (Historical)', best_sharpe]
            st.metric("⚖️ Danh mục có Sharpe tốt nhất", best_sharpe, f"{best_sharpe_val:.2f}")
        
        with col3:
            lowest_risk = summary_table.loc['Rủi ro Năm (Annualized)'].idxmin()
            lowest_risk_val = summary_table.loc['Rủi ro Năm (Annualized)', lowest_risk]
            st.metric("🛡️ Danh mục có Risk thấp nhất", lowest_risk, f"{lowest_risk_val:.2%}")
        
        st.divider()
        
        st.subheader(f"So sánh Hiệu quả Tăng trưởng (Từ {st.session_state.start_time_str})")
        fig_backtest = px.line(
            st.session_state.all_cumulative_df, 
            title=f'So sánh Hiệu quả Tăng trưởng (Từ {st.session_state.start_time_str})'
        )
        fig_backtest.update_layout(
            template='plotly_dark', 
            yaxis_title='Giá trị Danh mục (Bắt đầu từ 1.0)', 
            legend_title='Danh mục',
            yaxis_tickformat='.2f',
            height=600,
            hovermode='x unified'
        )
        st.plotly_chart(fig_backtest, use_container_width=True)
        
        # THÊM: Biểu đồ Drawdown
        st.subheader("Phân tích Drawdown (Mức sụt giảm)")
        drawdown_data = {}
        for col in st.session_state.all_cumulative_df.columns:
            cumulative = st.session_state.all_cumulative_df[col]
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            drawdown_data[col] = drawdown
        
        drawdown_df = pd.DataFrame(drawdown_data)
        fig_drawdown = px.line(
            drawdown_df,
            title='Phân tích Drawdown theo thời gian'
        )
        fig_drawdown.update_layout(
            template='plotly_dark',
            yaxis_title='Drawdown',
            yaxis_tickformat='.1%',
            height=500,
            hovermode='x unified'
        )
        fig_drawdown.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        st.plotly_chart(fig_drawdown, use_container_width=True)
        
        st.divider()
        
        st.subheader("Phân tích Chi tiết Hiệu suất Backtest")
        port_names = summary_table.columns
        fig_metrics = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=("So sánh Lợi nhuận", "So sánh Rủi ro", "So sánh Tỷ lệ (Sharpe)")
        )
        # 1. Lợi nhuận
        return_metrics = ['Tổng Lợi nhuận (Cumulative)', 'Lợi nhuận TB Năm (Annualized)']
        for metric in return_metrics:
            fig_metrics.add_trace(go.Bar(
                x=port_names, y=summary_table.loc[metric], text=summary_table.loc[metric],
                texttemplate='%{y:.2%}', name=metric
            ), row=1, col=1)
        # 2. Rủi ro
        risk_metrics = ['Rủi ro Năm (Annualized)', 'Mức sụt giảm Tối đa (Max Drawdown)']
        for metric in risk_metrics:
            fig_metrics.add_trace(go.Bar(
                x=port_names, y=summary_table.loc[metric], text=summary_table.loc[metric],
                texttemplate='%{y:.2%}', name=metric
            ), row=2, col=1)
        # 3. Sharpe
        sharpe_metric = 'Chỉ số Sharpe (Historical)'
        fig_metrics.add_trace(go.Bar(
            x=port_names, y=summary_table.loc[sharpe_metric], text=summary_table.loc[sharpe_metric],
            texttemplate='%{y:.2f}', name=sharpe_metric, marker_color='cyan'
        ), row=3, col=1)
        
        fig_metrics.update_layout(height=1000, template='plotly_dark', barmode='group')
        fig_metrics.update_yaxes(title_text='Lợi nhuận', tickformat='.0%', row=1, col=1)
        fig_metrics.update_yaxes(title_text='Rủi ro', tickformat='.0%', row=2, col=1)
        fig_metrics.update_yaxes(title_text='Tỷ lệ', tickformat='.2f', row=3, col=1)
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Tab 4: Phân tích Rủi ro & Đa dạng
    with tab4:
        st.header("🛡️ Phân tích Rủi ro & Đa dạng hóa")
        
        st.info(f"""
        **💡 Nguyên tắc "Không bỏ hết trứng vào 1 giỏ":**
        - Tỷ trọng tối đa mỗi cổ phiếu: **{MAX_WEIGHT*100:.0f}%**
        - Số cổ phiếu tối thiểu: **{min_stocks} cổ phiếu**
        - Mục tiêu: Giảm rủi ro tập trung, tăng tính ổn định của danh mục
        """)
        
        st.divider()
        
        # THÊM: Herfindahl Index - Chỉ số tập trung
        st.subheader("Chỉ số Tập trung (Herfindahl Index)")
        st.markdown("""
        **Herfindahl Index** đo lường mức độ tập trung của danh mục:
        - Giá trị gần 1: Tập trung cao (rủi ro)
        - Giá trị gần 0: Đa dạng hóa tốt (ít rủi ro)
        """)
        
        herfindahl_data = {}
        for col in st.session_state.optimal_weights_df.columns:
            weights = st.session_state.optimal_weights_df[col]
            herfindahl = (weights ** 2).sum()
            herfindahl_data[col] = herfindahl
        
        herfindahl_df = pd.DataFrame({
            'Danh mục': list(herfindahl_data.keys()),
            'Herfindahl Index': list(herfindahl_data.values())
        })
        
        fig_herfindahl = px.bar(
            herfindahl_df, x='Danh mục', y='Herfindahl Index',
            title='Chỉ số Herfindahl - Mức độ Tập trung Danh mục',
            text='Herfindahl Index',
            color='Herfindahl Index',
            color_continuous_scale='RdYlGn_r'
        )
        fig_herfindahl.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_herfindahl.update_layout(template='plotly_dark', height=500, showlegend=False)
        fig_herfindahl.add_hline(y=1/min_stocks, line_dash="dash", line_color="yellow",
                                annotation_text=f"Phân bổ đều {min_stocks} CP")
        st.plotly_chart(fig_herfindahl, use_container_width=True)
        
        st.divider()
        
        # THÊM: Contribution to Risk (VaR)
        st.subheader("Đóng góp Rủi ro của từng Cổ phiếu")
        selected_portfolio_risk = st.selectbox(
            "Chọn danh mục để phân tích:",
            options=st.session_state.optimal_weights_df.columns.tolist(),
            key='risk_analysis'
        )
        
        weights = st.session_state.optimal_weights_df[selected_portfolio_risk].values
        cov_matrix = st.session_state.cov_matrix
        
        # Tính Marginal Contribution to Risk
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_std
        contrib_to_risk = weights * marginal_contrib
        contrib_to_risk_pct = contrib_to_risk / contrib_to_risk.sum()
        
        risk_contrib_df = pd.DataFrame({
            'Cổ phiếu': st.session_state.optimal_weights_df.index,
            'Tỷ trọng': weights,
            'Đóng góp Rủi ro (%)': contrib_to_risk_pct
        }).sort_values('Đóng góp Rủi ro (%)', ascending=False).head(15)
        
        fig_risk_contrib = go.Figure()
        fig_risk_contrib.add_trace(go.Bar(
            x=risk_contrib_df['Cổ phiếu'],
            y=risk_contrib_df['Tỷ trọng'],
            name='Tỷ trọng',
            marker_color='lightblue',
            yaxis='y',
            offsetgroup=1
        ))
        fig_risk_contrib.add_trace(go.Bar(
            x=risk_contrib_df['Cổ phiếu'],
            y=risk_contrib_df['Đóng góp Rủi ro (%)'],
            name='Đóng góp Rủi ro',
            marker_color='salmon',
            yaxis='y',
            offsetgroup=2
        ))
        
        fig_risk_contrib.update_layout(
            title=f'So sánh Tỷ trọng vs Đóng góp Rủi ro - {selected_portfolio_risk}',
            xaxis_title='Cổ phiếu',
            yaxis_title='Giá trị (%)',
            template='plotly_dark',
            height=600,
            barmode='group',
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig_risk_contrib, use_container_width=True)
        
        st.markdown("""
        **Giải thích:**
        - **Tỷ trọng**: Tỷ lệ % vốn đầu tư vào mỗi cổ phiếu
        - **Đóng góp Rủi ro**: % rủi ro mà mỗi cổ phiếu đóng góp vào tổng rủi ro danh mục
        - Nếu Đóng góp Rủi ro >> Tỷ trọng → Cổ phiếu này có tương quan cao với các cổ phiếu khác
        """)
        
        st.divider()
        
        # THÊM: Effective Number of Assets
        st.subheader("Số lượng Cổ phiếu Hiệu quả (ENB)")
        st.markdown("""
        **ENB (Effective Number of Bets)** = 1 / Herfindahl Index  
        Đo lường số lượng cổ phiếu "thực sự độc lập" trong danh mục.
        """)
        
        enb_data = {}
        for col in st.session_state.optimal_weights_df.columns:
            weights = st.session_state.optimal_weights_df[col]
            herfindahl = (weights ** 2).sum()
            enb = 1 / herfindahl if herfindahl > 0 else 0
            enb_data[col] = {
                'ENB': enb,
                'Số CP thực tế': st.session_state.num_stocks[col],
                'Hiệu quả Đa dạng': enb / st.session_state.num_stocks[col] if st.session_state.num_stocks[col] > 0 else 0
            }
        
        enb_df = pd.DataFrame(enb_data).T
        enb_df_styled = enb_df.style.format({
            'ENB': '{:.2f}',
            'Số CP thực tế': '{:.0f}',
            'Hiệu quả Đa dạng': '{:.1%}'
        }).background_gradient(cmap='RdYlGn', subset=['Hiệu quả Đa dạng'])
        
        st.dataframe(enb_df_styled, use_container_width=True)
        
        st.markdown("""
        **Giải thích:**
        - **ENB**: Số cổ phiếu có trọng số bằng nhau tương đương với danh mục hiện tại
        - **Hiệu quả Đa dạng**: Tỷ lệ ENB / Số CP thực tế (càng cao càng tốt, tối đa 100%)
        - Hiệu quả 100% = Phân bổ hoàn toàn đều, < 50% = Tập trung cao
        """)
        
        st.divider()
        
        # THÊM: Risk-Return Scatter của từng cổ phiếu
        st.subheader("Ma trận Rủi ro - Lợi nhuận từng Cổ phiếu")
        
        individual_stats = pd.DataFrame({
            'Return': st.session_state.expected_returns,
            'Risk': np.sqrt(np.diag(st.session_state.cov_matrix))
        })
        individual_stats['Sharpe'] = (individual_stats['Return'] - RISK_FREE_RATE) / individual_stats['Risk']
        
        fig_scatter = px.scatter(
            individual_stats, 
            x='Risk', 
            y='Return',
            text=individual_stats.index,
            color='Sharpe',
            color_continuous_scale='RdYlGn',
            title='Ma trận Rủi ro - Lợi nhuận của từng Cổ phiếu trong VN30',
            size=abs(individual_stats['Sharpe']),
            size_max=15
        )
        fig_scatter.update_traces(textposition='top center', textfont_size=8)
        fig_scatter.update_layout(
            template='plotly_dark',
            height=700,
            xaxis_tickformat='.1%',
            yaxis_tickformat='.1%',
            xaxis_title='Rủi ro (Độ lệch chuẩn)',
            yaxis_title='Lợi nhuận Kỳ vọng'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tab 5: Sector Analysis & ML
    with tab5:
        st.header("🏢 Phân tích Phân bổ theo Ngành & Machine Learning")
        
        # === PHẦN 1: SECTOR ANALYSIS ===
        st.subheader("Phân tích Phân bổ theo Ngành")
        
        sector_allocation = get_sector_allocation(st.session_state.optimal_weights_df)
        
        # Hiển thị bảng
        st.dataframe(
            sector_allocation.style.format("{:.2%}").background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        # Biểu đồ cột so sánh
        sector_long = sector_allocation.reset_index().melt(
            id_vars='index',
            var_name='Danh mục',
            value_name='Tỷ trọng'
        )
        sector_long.rename(columns={'index': 'Ngành'}, inplace=True)
        
        fig_sector = px.bar(
            sector_long,
            x='Danh mục',
            y='Tỷ trọng',
            color='Ngành',
            title='Phân bổ theo Ngành - So sánh 3 Danh mục',
            text_auto='.1%'
        )
        fig_sector.update_layout(template='plotly_dark', yaxis_tickformat='.0%', height=500)
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Biểu đồ tròn cho từng danh mục
        col1, col2, col3 = st.columns(3)
        
        for idx, (col, portfolio) in enumerate(zip([col1, col2, col3], sector_allocation.columns)):
            with col:
                sector_data = sector_allocation[portfolio]
                sector_data = sector_data[sector_data > 0.001]
                
                fig_sector_pie = go.Figure(data=[go.Pie(
                    labels=sector_data.index,
                    values=sector_data.values,
                    hole=0.3
                )])
                fig_sector_pie.update_layout(
                    title=portfolio,
                    template='plotly_dark',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_sector_pie, use_container_width=True)
        
        # Phân tích đa dạng hóa ngành
        st.subheader("Chỉ số Đa dạng hóa theo Ngành")
        
        sector_diversity = {}
        for portfolio in sector_allocation.columns:
            sectors = sector_allocation[portfolio]
            sectors = sectors[sectors > 0.001]
            hhi = (sectors ** 2).sum()
            enb = 1 / hhi if hhi > 0 else 0
            sector_diversity[portfolio] = {
                'Số ngành': len(sectors),
                'HHI (Ngành)': hhi,
                'ENB (Ngành)': enb,
                'Ngành lớn nhất': sectors.idxmax(),
                'Tỷ trọng lớn nhất': sectors.max()
            }
        
        sector_div_df = pd.DataFrame(sector_diversity).T
        st.dataframe(
            sector_div_df.style.format({
                'Số ngành': '{:.0f}',
                'HHI (Ngành)': '{:.3f}',
                'ENB (Ngành)': '{:.2f}',
                'Tỷ trọng lớn nhất': '{:.2%}'
            }),
            use_container_width=True
        )
        
        st.divider()
        
        # === PHẦN 2: MACHINE LEARNING ===
        st.header("🤖 Machine Learning - Phân tích Nâng cao")
        
        # === PHẦN 1: DỰ ĐOÁN RETURNS ===
        st.subheader("Dự đoán Lợi nhuận bằng Random Forest")
        
        with st.spinner("⏳ Đang huấn luyện mô hình Machine Learning..."):
            predictions, feature_importance = ml_predict_returns(
                st.session_state.returns_df,
                st.session_state.price_pivot
            )
        
        if predictions:
            # Chọn cổ phiếu để xem
            selected_ticker_ml = st.selectbox(
                "Chọn cổ phiếu để xem dự đoán:",
                options=list(predictions.keys())
            )
            
            pred_data = predictions[selected_ticker_ml]
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Độ chính xác (R² Score)", f"{pred_data['score']:.3f}")
            with col2:
                mae = np.mean(np.abs(pred_data['actual'] - pred_data['predicted']))
                st.metric("📊 MAE", f"{mae:.4f}")
            with col3:
                rmse = np.sqrt(np.mean((pred_data['actual'] - pred_data['predicted'])**2))
                st.metric("📉 RMSE", f"{rmse:.4f}")
            
            # Biểu đồ so sánh Actual vs Predicted
            comparison_ml = pd.DataFrame({
                'Thực tế': pred_data['actual'],
                'Dự đoán': pred_data['predicted']
            })
            
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(
                y=comparison_ml['Thực tế'],
                mode='lines',
                name='Thực tế',
                line=dict(color='cyan')
            ))
            fig_ml.add_trace(go.Scatter(
                y=comparison_ml['Dự đoán'],
                mode='lines',
                name='Dự đoán',
                line=dict(color='orange', dash='dash')
            ))
            fig_ml.update_layout(
                title=f'So sánh Returns Thực tế vs Dự đoán - {selected_ticker_ml}',
                template='plotly_dark',
                yaxis_title='Returns',
                xaxis_title='Thời gian (Test Set)',
                height=500
            )
            st.plotly_chart(fig_ml, use_container_width=True)
            
            # Feature Importance
            st.subheader("Độ quan trọng của Features")
            fi_df = feature_importance[selected_ticker_ml].sort_values(ascending=False)
            
            fig_fi = px.bar(
                x=fi_df.values,
                y=fi_df.index,
                orientation='h',
                title=f'Feature Importance - {selected_ticker_ml}',
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            fig_fi.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.info("""
            **Giải thích Features:**
            - **return_lag1/2/3**: Lợi nhuận 1, 2, 3 ngày trước
            - **return_ma5/20**: Moving average 5, 20 ngày
            - **volatility_20**: Độ biến động 20 ngày
            """)
        else:
            st.warning("Không đủ dữ liệu để huấn luyện mô hình ML")
        
        st.divider()
        
        # === PHẦN 2: CLUSTERING ===
        st.subheader("Phân nhóm Cổ phiếu (K-Means Clustering)")
        
        n_clusters = st.slider(
            "Số nhóm (clusters):",
            min_value=2, max_value=5, value=3, step=1
        )
        
        with st.spinner("⏳ Đang phân nhóm cổ phiếu..."):
            cluster_result = cluster_stocks(st.session_state.returns_df, n_clusters)
        
        # Hiển thị bảng
        st.dataframe(
            cluster_result.style.format({
                'mean_return': '{:.4f}',
                'volatility': '{:.4f}',
                'sharpe': '{:.2f}',
                'skewness': '{:.2f}',
                'kurtosis': '{:.2f}'
            }).background_gradient(cmap='viridis', subset=['Cluster']),
            use_container_width=True
        )
        
        # Biểu đồ scatter 3D
        fig_cluster = px.scatter_3d(
            cluster_result,
            x='mean_return',
            y='volatility',
            z='sharpe',
            color='Cluster',
            text='Ticker',
            title='Phân nhóm Cổ phiếu theo Risk-Return Profile',
            labels={
                'mean_return': 'Return TB',
                'volatility': 'Volatility',
                'sharpe': 'Sharpe Ratio'
            },
            color_continuous_scale='viridis'
        )
        fig_cluster.update_traces(textposition='top center', textfont_size=8)
        fig_cluster.update_layout(template='plotly_dark', height=700)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Phân tích từng cluster
        st.subheader("Đặc điểm từng Nhóm")
        
        for cluster_id in range(n_clusters):
            cluster_data = cluster_result[cluster_result['Cluster'] == cluster_id]
            
            with st.expander(f"🔹 Nhóm {cluster_id} ({len(cluster_data)} cổ phiếu)"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Return TB", f"{cluster_data['mean_return'].mean():.4f}")
                with col2:
                    st.metric("Volatility TB", f"{cluster_data['volatility'].mean():.4f}")
                with col3:
                    st.metric("Sharpe TB", f"{cluster_data['sharpe'].mean():.2f}")
                with col4:
                    st.metric("Số cổ phiếu", len(cluster_data))
                
                st.write("**Danh sách cổ phiếu:**", ", ".join(cluster_data['Ticker'].tolist()))
        
        st.info("""
        **Ứng dụng Clustering:**
        - Xác định các cổ phiếu có đặc tính tương tự
        - Đa dạng hóa bằng cách chọn cổ phiếu từ các nhóm khác nhau
        - Hiểu rõ hơn về cấu trúc thị trường VN30
        """)
        
        st.divider()
        
        # === PHẦN 3: KHUYẾN NGHỊ ===
        st.subheader("Khuyến nghị Dựa trên Machine Learning")
        
        # Top cổ phiếu theo ML score
        if predictions:
            ml_scores = {ticker: data['score'] for ticker, data in predictions.items()}
            top_ml = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            st.success("**Top 5 cổ phiếu dự đoán tốt nhất (ML Score cao):**")
            for i, (ticker, score) in enumerate(top_ml, 1):
                st.write(f"{i}. **{ticker}**: R² Score = {score:.3f}")
        
        # Top cổ phiếu theo Sharpe trong mỗi cluster
        st.warning("**Khuyến nghị Đa dạng hóa theo Cluster:**")
        for cluster_id in range(n_clusters):
            cluster_data = cluster_result[cluster_result['Cluster'] == cluster_id]
            best_in_cluster = cluster_data.nlargest(1, 'sharpe')
            if not best_in_cluster.empty:
                ticker = best_in_cluster.iloc[0]['Ticker']
                sharpe = best_in_cluster.iloc[0]['sharpe']
                st.write(f"- **Nhóm {cluster_id}**: Chọn **{ticker}** (Sharpe = {sharpe:.2f})")

# [SỬA 2] Xóa bỏ dấu } bị lỗi
# }