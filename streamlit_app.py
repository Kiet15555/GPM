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

# --- Cấu hình Trang & Các thiết lập ban đầu ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Tối ưu Danh mục VN30 Pro")
st.title("📈 Ứng dụng Phân tích & Tối ưu hóa Danh mục VN30 (Bản Full)")

# Set theme mặc định cho Plotly
pio.templates.default = "plotly_dark"


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

    # 5. Chạy Tối ưu hóa
    with st.spinner("⏳ (5/7) Đang chạy Tối ưu hóa..."):
        num_assets = len(st.session_state.expected_returns)
        args = (st.session_state.expected_returns, st.session_state.cov_matrix, RISK_FREE_RATE)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        # 1. Min Risk
        min_vol_guess_weights = sim_data_df.loc[sim_data_df['Risk'].idxmin()].values[3:3+num_assets]
        opt_min_vol = minimize(minimize_portfolio_risk, min_vol_guess_weights, args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)
        min_vol_weights = opt_min_vol.x

        # 2. Max Sharpe
        max_sharpe_guess_weights = sim_data_df.loc[sim_data_df['Sharpe'].idxmax()].values[3:3+num_assets]
        opt_max_sharpe = minimize(minimize_negative_sharpe, max_sharpe_guess_weights, args=args,
                                  method='SLSQP', bounds=bounds, constraints=constraints)
        max_sharpe_weights = opt_max_sharpe.x
        
        # 3. Max Return
        max_ret_weights = np.zeros(num_assets)
        max_ret_index = st.session_state.expected_returns.argmax()
        max_ret_weights[max_ret_index] = 1.0

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
    
    tab1, tab2, tab3 = st.tabs(["Đường biên Hiệu quả & Tỷ trọng", "Kết quả Backtest", "Dữ liệu Thô & Tương quan"])

    with tab1:
        st.header("1. Phân bổ Tỷ trọng & Đường biên Hiệu quả")
        
        # --- PHẦN 1: BẢNG SỐ LIỆU ---
        st.subheader("Phân bổ Tỷ trọng Tối ưu")
        df_weights = st.session_state.optimal_weights_df
        df_weights_styled = df_weights[(df_weights > 0.001).any(axis=1)].style.format("{:.2%}")
        st.dataframe(df_weights_styled, use_container_width=True)
        
        st.divider()
        
        # --- PHẦN 2: BIỂU ĐỒ TỶ TRỌNG ---
        st.subheader("Biểu đồ Tỷ trọng")
        df_plot = df_weights[(df_weights > 0.001).any(axis=1)].copy()
        
        df_plot_long = df_plot.reset_index().melt(
            id_vars='ticker', 
            var_name='Danh mục', 
            value_name='Tỷ trọng'
        )
        df_plot_long.rename(columns={'ticker': 'Mã CP'}, inplace=True)
        
        fig_bars = px.bar(
            df_plot_long, x='Danh mục', y='Tỷ trọng', color='Mã CP',
            text_auto='.2%',
            title='Phân bổ Tỷ trọng Tối ưu theo 3 Khẩu vị Rủi ro'
        )
        fig_bars.update_layout(template='plotly_dark', yaxis_tickformat='.0%', height=500)
        st.plotly_chart(fig_bars, use_container_width=True)
            
        st.divider()

        # --- PHẦN 3: BIỂU ĐỒ ĐƯỜNG BIÊN HIỆU QUẢ ---
        st.subheader("Đường biên Hiệu quả Toàn diện (có CAL)")
        
        # FIX: THÊM render_mode='svg' ĐỂ SỬA LỖI WEBGL
        sim_data_df = st.session_state.sim_data_df
        fig = px.scatter(
            sim_data_df, x='Risk', y='Return', color='Sharpe',
            color_continuous_scale='Viridis',
            hover_data={col: ':.2%' for col in sim_data_df.columns if col not in ['Risk', 'Return', 'Sharpe']} | {'Risk': ':.2%','Return': ':.2%','Sharpe': ':.2f'},
            title=f'Đường biên Hiệu quả - {N_SIMULATIONS} danh mục (Rf={RISK_FREE_RATE:.1%})',
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
        
        fig.add_trace(go.Scatter(x=[stats_min_vol[1]], y=[stats_min_vol[0]], mode='markers', marker=dict(color='white', size=15, symbol='star', line=dict(color='black', width=2)), name='Bảo thủ (Min Risk)'))
        fig.add_trace(go.Scatter(x=[stats_max_sharpe[1]], y=[stats_max_sharpe[0]], mode='markers', marker=dict(color='cyan', size=15, symbol='star', line=dict(color='black', width=2)), name='Cân bằng (Max Sharpe)'))
        fig.add_trace(go.Scatter(x=[stats_max_ret[1]], y=[stats_max_ret[0]], mode='markers', marker=dict(color='red', size=15, symbol='star', line=dict(color='black', width=2)), name='Mạo hiểm (Max Return)'))
        
        # Vẽ Đường CAL
        sharpe_risk = stats_max_sharpe[1]
        sharpe_return = stats_max_sharpe[0]
        x_cal = [0, sharpe_risk * 1.5] 
        y_cal = [RISK_FREE_RATE, (sharpe_return - RISK_FREE_RATE) / (sharpe_risk + 1e-9) * (sharpe_risk * 1.5) + RISK_FREE_RATE]
        fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', line=dict(color='lime', width=2, dash='dash'), name='Đường Phân bổ Vốn (CAL)'))

        fig.update_layout(
            height=800,
            xaxis_tickformat='.1%', yaxis_tickformat='.1%',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(b=100)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("2. Kết quả Backtest")
        
        st.subheader("Bảng Tổng kết Chỉ số Hiệu suất")
        summary_table = st.session_state.summary_table
        percent_rows = summary_table.index.difference(['Chỉ số Sharpe (Historical)'])
        number_row = pd.Index(['Chỉ số Sharpe (Historical)'])
        styler = summary_table.style
        styler.format('{:,.2%}', subset=(percent_rows, slice(None)))
        styler.format('{:,.2f}', subset=(number_row, slice(None)))
        st.dataframe(styler, use_container_width=True)
        
        st.subheader(f"So sánh Hiệu quả Tăng trưởng (Từ {st.session_state.start_time_str})")
        fig_backtest = px.line(
            st.session_state.all_cumulative_df, 
            title=f'So sánh Hiệu quả Tăng trưởng (Từ {st.session_state.start_time_str})'
        )
        fig_backtest.update_layout(
            template='plotly_dark', 
            yaxis_title='Giá trị Danh mục (Bắt đầu từ 1.0)', 
            legend_title='Danh mục',
            yaxis_tickformat='.2f'
        )
        st.plotly_chart(fig_backtest, use_container_width=True)
        
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
            texttemplate='%{y:.2f}', name=sharpe_metric
        ), row=3, col=1)
        
        fig_metrics.update_layout(height=1000, template='plotly_dark', barmode='group')
        fig_metrics.update_yaxes(title_text='Lợi nhuận', tickformat='.0%', row=1, col=1)
        fig_metrics.update_yaxes(title_text='Rủi ro', tickformat='.0%', row=2, col=1)
        fig_metrics.update_yaxes(title_text='Tỷ lệ', tickformat='.2f', row=3, col=1)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
    with tab3:
        st.header("3. Dữ liệu Thô & Phân tích Tương quan")
        
        st.subheader("Heatmap Ma trận Tương quan")
        returns_df = st.session_state.returns_df
        correlation_matrix = returns_df.corr()
        labels = correlation_matrix.columns
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values, x=labels, y=labels,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            hoverongaps=False
        ))
        fig_heatmap.update_layout(
            title='Heatmap Ma trận Tương quan (VN30)', template='plotly_dark',
            height=700, width=800,
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.subheader("Dữ liệu Giá Đóng cửa (Pivot)")
        st.dataframe(st.session_state.price_pivot.tail())
        
        st.subheader("Dữ liệu Tỷ suất sinh lời (Hàng ngày)")
        st.dataframe(st.session_state.returns_df.tail())
        
        st.subheader("Dữ liệu Thô (Tải về)")
        st.dataframe(st.session_state.raw_data.tail())