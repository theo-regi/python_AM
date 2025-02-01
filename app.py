import os

# Get the absolute path to the app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for assets
ASSETS_DIR = os.path.join(APP_DIR, "assets")
STYLE_PATH = os.path.join(ASSETS_DIR, "style.css")
LOGO_PATH = os.path.join(ASSETS_DIR, "dauphine_logo.png")

from scripts import prerequisite_check
from scripts import function as fct
from scripts import user_parameters as params
from scripts import market_views as mkv

prerequisite_check.global_check()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# SETTING UP THE PAGE
# ============================================================
st.set_page_config(page_title="Asset Management App", layout="wide")
# Load the custom CSS
def load_css(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(STYLE_PATH)
st.sidebar.image(LOGO_PATH, use_container_width =True)

# Sidebar for user inputs
st.sidebar.markdown("### Asset Management Application")
st.sidebar.header("User Inputs")

# ============================================================
# PARAMETERS DEFINED BY USER
# ============================================================
# Data source selection
st.sidebar.markdown("### Data Source Selection")
#data_source = st.sidebar.radio("Data Source", ["Excel", "Bloomberg"], index=1 if params.get_user_choice() else 0)
data_source="Excel"

# General parameters
st.sidebar.markdown("### General Parameters")
frequency = st.sidebar.number_input("Frequency (days/year)", value=params.get_frequency())
n_portfolios = st.sidebar.slider("Number of Portfolios", min_value=100, max_value=50000, value=params.get_n_portfolios(), step=100)
start_date = st.sidebar.text_input("Start Date (YYYYMMDD)", value=params.get_stend_date()[0])
end_date = st.sidebar.text_input("End Date (YYYYMMDD)", value=params.get_stend_date()[1])

# Portfolio constraints
st.sidebar.markdown("### Portfolio Constraints")
vol_target = st.sidebar.slider("Volatility Target", min_value=0.01, max_value=0.1, value=params.get_target_vol(), step=0.01)

# Black-Litterman parameters
st.sidebar.markdown("### Black-Litterman Risk Aversion")
tau = st.sidebar.slider("Investors sensibility to market views (Tau)", min_value=0.000, max_value=1.000, value=params.get_tau(), step=0.005)
risk_aversions = params.get_risk_aversions()
adjusted_risk_aversions = {}
for profile, default_value in risk_aversions.items():
    adjusted_value = st.sidebar.slider(
        f"{profile} Risk Aversion",
        min_value=0.1,
        max_value=10.0,
        value=float(default_value),
        step=0.1
    )
    adjusted_risk_aversions[profile] = adjusted_value

#============================================================
# MAIN PROGRAM SCRIPT FOR THE PYTHON ASSET MANAGEMENT PROJECT
#============================================================
# Main app
st.title("Asset Management App")
st.markdown("### Visualize and Manage Asset Portfolios")

# ================================================
# DATA IMPORTATION FROM BBG OR EXCEL = USER CHOICE
# ================================================
# Data import
if data_source == "Bloomberg":
    st.info("Using Bloomberg data (requires Bloomberg API).")
    names = params.get_names()  # Ensure `names` is defined
    tickers = params.get_tickers()
    prices = fct.blomberg_importation(names, tickers, start_date, end_date)
else:
    st.info("Using data from 'Prices Frontier.xlsx'.")
    prices = fct.excel_importation('data/Prices Frontier.xlsx')

names = params.get_names()

# Option to download raw data
st.write("### Download Raw Data")
if st.button("Download Raw Data"):
    raw_data_file = "raw_data.xlsx"
    prices.to_excel(raw_data_file)
    with open(raw_data_file, "rb") as f:
        st.download_button("Download Excel File", f, file_name="raw_data.xlsx")

# ================================================================
# RETURNS AND TRACKS CALCULATION -> PLOT ASSETS CUMULATIVE RETURNS
# ================================================================
# Calculate returns and tracks
st.write("### Returns and Tracks")
returns = fct.calculate_returns(prices)
tracks = fct.calculate_tracks_multidim(returns)
# Plot tracks
st.subheader("Cumulative Returns (Tracks)")
fig_tracks = go.Figure()

for col in tracks.columns:
    fig_tracks.add_trace(go.Scatter(x=tracks.index, y=tracks[col], mode='lines', name=col))

fig_tracks.update_layout(
    title="Cumulative Returns",
    xaxis_title="Time",
    yaxis_title="Portfolio Value",
    yaxis=dict(type="linear"),  # Auto-adapting scale
    legend_title="Assets"
)
st.plotly_chart(fig_tracks)

# Download options for returns and tracks
col1, col2 = st.columns(2)
with col1:
    st.write("Download Returns")
    if st.button("Download Returns"):
        returns_file = "returns.xlsx"
        returns.to_excel(returns_file)
        with open(returns_file, "rb") as f:
            st.download_button("Download Excel File", f, file_name="returns.xlsx")

with col2:
    st.write("Download Tracks")
    if st.button("Download Tracks"):
        tracks_file = "tracks.xlsx"
        tracks.to_excel(tracks_file)
        with open(tracks_file, "rb") as f:
            st.download_button("Download Excel File", f, file_name="tracks.xlsx")
            
# ================================================================
#           BASIC CALCULATIONS FOR ALL THE NEXT PARTS
# ================================================================
#Getting annualized returns, annualized volatility, covariance matrix and sharpe ratio of each asset
an_returns = fct.an_returns_from_tracks(tracks, frequency)
an_volatilities = fct.an_vol_from_returns(returns, frequency)
sharpe_ratios = fct.sharpe_ratio_with_cash(an_returns, an_volatilities)
cov_matrix = fct.cov_matrix(returns, frequency)

#Calculating max drawdowns and expected returns
max_drawdowns = []
for name in names:
    max_drawdowns.append(fct.max_DD(tracks[name]))
exp_returns = fct.expected_returns(an_returns, names, an_volatilities, max_drawdowns)    

# =========================================
# PORTFOLIO GENERATION FOR OPTIMAL FRONTIER
# =========================================
weights, mean_variance_pairs, fig = fct.generate_portfolios(n_portfolios, returns, exp_returns, an_returns, an_volatilities, frequency)

# ===============================================================
# MAXIMIZATION OF PORTFOLIO RETURNS FOR A GIVEN VOLATILITY TARGET
# ===============================================================
# Constraints:
# ------------
init_weights = params.get_init_weights()
max_weights = params.get_max_weights()
target_vol = params.get_target_vol()

#Maximization:
constraint_weights = fct.constraint_weights(weights)
constraint_weights_bounds = fct.constraint_weights_bounds(weights, init_weights)
constraint_weight_by_asset = fct.constraint_weights_by_asset(weights, max_weights)

#MAXIMUM RETURNS PORTFOLIO FOR TARGET VOL
weights_MR4 = fct.maxReturn(returns, target_vol, frequency, weights, constraint_weights, constraint_weight_by_asset, constraint_weights_bounds, init_weights)
return_MR4 = fct.ptfReturn(returns, weights_MR4, frequency)
vol_MR4 = fct.ptfVol(returns, weights_MR4, frequency)
st.subheader("Target Volatility returns and resulting volatility")
st.write('Returns for target vol: ', "{:.2f}%".format(return_MR4[0]*100))
st.write('Resulting volatility: ', "{:.2f}%".format(vol_MR4*100))

# ===========================
# FINDING MAX SHARPE PORFOLIO
# ===========================
optimal_weights = fct.maxSharpe(returns, frequency, an_returns,  weights, constraint_weights, constraint_weight_by_asset, constraint_weights_bounds, init_weights)
ptf_opti_returns = fct.ptfReturn(returns, optimal_weights, frequency)
vol_optimal_ptf = fct.ptfVol(returns, optimal_weights, frequency)
opti_returns = (returns * optimal_weights).sum(axis=1)
opti_vol = fct.an_vol_from_returns(opti_returns, frequency)
cash_return = fct.cash_an_returns_from_tracks(tracks, returns, frequency)
optimal_perf = fct.an_returns_from_tracks(((1+tracks.tail(1)) * optimal_weights).sum(axis=1), frequency) #optimal_perf = ((1+tracks.tail(1)) * optimal_weights).sum(axis=1)**(252/len(returns))-1
optimal_returns_sharpe = (optimal_perf - cash_return)/ opti_vol
st.subheader("Optimal Portfolio Weights to Maximize Sharpe Ratio:")
for names, weights in zip(names, optimal_weights):
    st.write(f"{names}: {weights:.2%}")
    
# ================================
# CONSTRUCTING CAPITAL MARKET LINE
# ================================
# Assuming risk-free rate is the return of the 'Cash' asset
risk_free_rate = exp_returns[0]
max_sharpe_return, max_sharpe_volatility, fig = fct.capital_market_line_constructor(mean_variance_pairs, risk_free_rate)
st.subheader("Efficient Frontier")
st.plotly_chart(fig)

# =================================================
# BENCHMARK CONSTRUCTION FROM MAX SHARPE PORTFOLIO      KIND OF USELESS BECAUSE OF MONTHLY REBALANCING LATER
# =================================================
returns_ptf = (returns * optimal_weights).sum(axis=1)
returns_cash = returns['Cash']

bench_defensive = fct.benchmark_portfolio(0.5, 0.5, returns_cash, returns_ptf)
bench_balanced = fct.benchmark_portfolio(0.25, 0.75, returns_cash, returns_ptf)
bench_growth = fct.benchmark_portfolio(0, 1, returns_cash, returns_ptf)
st.subheader("Benchmark Performance")
fig_benchmarks = go.Figure()

fig_benchmarks.add_trace(go.Scatter(x=bench_defensive.index, y=bench_defensive, mode='lines', name="Defensive"))
fig_benchmarks.add_trace(go.Scatter(x=bench_balanced.index, y=bench_balanced, mode='lines', name="Balanced"))
fig_benchmarks.add_trace(go.Scatter(x=bench_growth.index, y=bench_growth, mode='lines', name="Growth"))

fig_benchmarks.update_layout(
    title="Benchmark Performance",
    xaxis_title="Time",
    yaxis_title="Value",
    yaxis=dict(type="linear"),  # Auto-adapting scale
    legend_title="Benchmarks"
)
st.plotly_chart(fig_benchmarks)

# ========================================
# REBALANCING PORFOLIOS ON A MONTHLY BASIS
# ========================================
#Get monthly returns for each assert and finding the maxsharpe ptf on montly rebalancing
monthly_returns = prices.resample('M').last().pct_change().dropna()
optimal_weights = fct.maxSharpe(monthly_returns, 1, an_returns,  weights, constraint_weights, constraint_weight_by_asset, constraint_weights_bounds, init_weights)
optimal_portfolio_value = fct.rebalance_portfolio(optimal_weights, monthly_returns)

#Get benchmarks with monthly rebalancing
returns_cash_monthly = monthly_returns['Cash']
monthly_optimal_ptf_returns= optimal_portfolio_value.pct_change().fillna(0)
bench_defensive_monthly = fct.benchmark_portfolio(0.5, 0.5, returns_cash_monthly, monthly_optimal_ptf_returns)
bench_balanced_monthly = fct.benchmark_portfolio(0.25, 0.75, returns_cash_monthly, monthly_optimal_ptf_returns)
bench_growth_monthly = fct.benchmark_portfolio(0, 1, returns_cash_monthly, monthly_optimal_ptf_returns)

st.subheader("Benchmark Performance (Rebalanced Monthly)")
fig_benchmarks_monthly = go.Figure()

fig_benchmarks_monthly.add_trace(go.Scatter(x=bench_defensive_monthly.index, y=bench_defensive_monthly, mode='lines', name="Defensive"))
fig_benchmarks_monthly.add_trace(go.Scatter(x=bench_balanced_monthly.index, y=bench_balanced_monthly, mode='lines', name="Balanced"))
fig_benchmarks_monthly.add_trace(go.Scatter(x=bench_growth_monthly.index, y=bench_growth_monthly, mode='lines', name="Growth"))

fig_benchmarks_monthly.update_layout(
    title="Benchmark Performance (Rebalanced Monthly)",
    xaxis_title="Time",
    yaxis_title="Value",
    yaxis=dict(type="linear"),  # Auto-adapting scale
    legend_title="Benchmarks"
)
st.plotly_chart(fig_benchmarks_monthly)


# =======================================
# SYSTEMATIC STATEGY APPLICATION: US JOBS
# =======================================
data_macro = fct.excel_importation('data/indicateurs.xlsx')
us_jobs_data = fct.prepare_us_jobs(data_macro)
indicator_us_jobs = fct.strategy_indicator(us_jobs_data, 1.3)

#--------------------------Portfolios with systematic strategy-----------------
# Portfolio to outperform growth benchmark
weights_exposed_equity = [0.0, 0.18, 0.17, 0.15, 0.0, 0.50, 0.0]
weights_exposed_oblig_E3M_high = [0.0, 0.0, 0.0, 0.90, 0.0, 0.10, 0]
returns_strat_jobs_growth, return_transaction_fees_growth = fct.rebalanced_returns(tracks, optimal_weights, indicator_us_jobs, weights_exposed_equity, weights_exposed_oblig_E3M_high, 0.3)
tracks_strat_jobs_growth = fct.calculate_tracks_unidim(returns_strat_jobs_growth.sum(axis=1))

#Portfolio outperforming balanced but not growth
weights_exposed_equity = [0.0, 0.18, 0.17, 0.40, 0.0, 0.25, 0.0]
weights_exposed_oblig_E3M_high = [0.0, 0.0, 0.0, 0.70, 0.20, 0.10, 0.0]
returns_strat_jobs_moderate, return_transaction_fees_moderate  = fct.rebalanced_returns(tracks, optimal_weights, indicator_us_jobs, weights_exposed_equity, weights_exposed_oblig_E3M_high, 0.3)
tracks_strat_jobs_moderate = fct.calculate_tracks_unidim(returns_strat_jobs_moderate.sum(axis=1))

#Portfolio outperforming defensive benchamrk but not balanced
weights_exposed_equity = [0.0, 0.12, 0.12, 0.65, 0.0, 0.11, 0.0]
weights_exposed_oblig_E3M_high = [0.0, 0.0, 0.0, 0.50, 0.50, 0.0, 0.0]
returns_strat_jobs_def, return_transaction_fees_def = fct.rebalanced_returns(tracks, optimal_weights, indicator_us_jobs, weights_exposed_equity, weights_exposed_oblig_E3M_high, 0.3)
tracks_strat_jobs_def =  fct.calculate_tracks_unidim(returns_strat_jobs_def.sum(axis=1))

st.subheader("Systematic Strategy Portfolio vs Benchmark Performance")
fig_sys_strategy = go.Figure()

fig_sys_strategy.add_trace(go.Scatter(x=bench_growth_monthly.index, y=bench_growth_monthly, mode='lines', name="Growth"))
fig_sys_strategy.add_trace(go.Scatter(x=bench_balanced_monthly.index, y=bench_balanced_monthly, mode='lines', name="Balanced"))
fig_sys_strategy.add_trace(go.Scatter(x=bench_defensive_monthly.index, y=bench_defensive_monthly, mode='lines', name="Defensive"))
fig_sys_strategy.add_trace(go.Scatter(x=tracks_strat_jobs_growth.index, y=tracks_strat_jobs_growth, mode='lines', name="Systematic US Jobs Strategy on Growth"))
fig_sys_strategy.add_trace(go.Scatter(x=tracks_strat_jobs_moderate.index, y=tracks_strat_jobs_moderate, mode='lines', name="Systematic US Jobs Strategy on Moderate"))
fig_sys_strategy.add_trace(go.Scatter(x=tracks_strat_jobs_def.index, y=tracks_strat_jobs_def, mode='lines', name="Systematic US Jobs Strategy on Defensive"))

fig_sys_strategy.update_layout(
    title="Systematic Strategy Portfolio vs Benchmark Performance",
    xaxis_title="Time",
    yaxis_title="Value",
    yaxis=dict(type="linear"),  # Auto-adapting scale
    legend_title="Portfolio"
)
st.plotly_chart(fig_sys_strategy)


#Strategies with annual and transaction fees.
tracks_strat_jobs_growth_with_fees = fct.apply_management_fees(returns_strat_jobs_growth.sum(axis=1), 0.015,return_transaction_fees_growth)
tracks_strat_jobs_moderate_with_fees = fct.apply_management_fees(returns_strat_jobs_moderate.sum(axis=1), 0.0125,return_transaction_fees_moderate)
tracks_strat_jobs_def_with_fees = fct.apply_management_fees(returns_strat_jobs_def.sum(axis=1), 0.01,return_transaction_fees_def)

st.subheader("Systematic Strategy Portfolio vs Benchmark Performance (with fees applied)")
fig_sys_strategy_fees = go.Figure()

fig_sys_strategy_fees.add_trace(go.Scatter(x=bench_growth_monthly.index, y=bench_growth_monthly, mode='lines', name="Growth"))
fig_sys_strategy_fees.add_trace(go.Scatter(x=bench_balanced_monthly.index, y=bench_balanced_monthly, mode='lines', name="Balanced"))
fig_sys_strategy_fees.add_trace(go.Scatter(x=bench_defensive_monthly.index, y=bench_defensive_monthly, mode='lines', name="Defensive"))
fig_sys_strategy_fees.add_trace(go.Scatter(x=tracks_strat_jobs_growth_with_fees.index, y=tracks_strat_jobs_growth_with_fees, mode='lines', name="Growth with Fees"))
fig_sys_strategy_fees.add_trace(go.Scatter(x=tracks_strat_jobs_moderate_with_fees.index, y=tracks_strat_jobs_moderate_with_fees, mode='lines', name="Moderate with Fees"))
fig_sys_strategy_fees.add_trace(go.Scatter(x=tracks_strat_jobs_def_with_fees.index, y=tracks_strat_jobs_def_with_fees, mode='lines', name="Defensive with Fees"))

fig_sys_strategy_fees.update_layout(
    title="Systematic Strategy Portfolio vs Benchmark Performance (with Fees Applied)",
    xaxis_title="Time",
    yaxis_title="Portfolio Value",
    yaxis=dict(type="linear"),  # Auto-adapting y-scale
    legend_title="Portfolio"
)

st.plotly_chart(fig_sys_strategy_fees)

# ============
# OUT OF SAMPLE
# ============
returns_strat_jobs_growth_out_of_sample, return_transaction_fees_growth = fct.rebalanced_returns_out_of_sample(tracks, optimal_weights, indicator_us_jobs, weights_exposed_equity, weights_exposed_oblig_E3M_high, 0.3)
tracks_strat_jobs_growth = fct.calculate_tracks_unidim(returns_strat_jobs_growth.sum(axis=1))

returns_strat_jobs_moderate_out_of_sample, return_transaction_fees_moderate  = fct.rebalanced_returns_out_of_sample(tracks, optimal_weights, indicator_us_jobs, weights_exposed_equity, weights_exposed_oblig_E3M_high, 0.3)
tracks_strat_jobs_moderate = fct.calculate_tracks_unidim(returns_strat_jobs_moderate.sum(axis=1))

returns_strat_jobs_def_out_of_sample, return_transaction_fees_def = fct.rebalanced_returns_out_of_sample(tracks, optimal_weights, indicator_us_jobs, weights_exposed_equity, weights_exposed_oblig_E3M_high, 0.3)
tracks_strat_jobs_def =  fct.calculate_tracks_unidim(returns_strat_jobs_def.sum(axis=1))

tracks_strat_jobs_growth_with_fees_out_of_sample = fct.apply_management_fees(returns_strat_jobs_growth.sum(axis=1), 0.015,return_transaction_fees_growth)
tracks_strat_jobs_moderate_with_fees_out_of_sample = fct.apply_management_fees(returns_strat_jobs_moderate.sum(axis=1), 0.0125,return_transaction_fees_moderate)
tracks_strat_jobs_def_with_fees_out_of_sample = fct.apply_management_fees(returns_strat_jobs_def.sum(axis=1), 0.01,return_transaction_fees_def)
information_ratio_growth = fct.ratio_information(tracks_strat_jobs_growth_with_fees.iloc[1:], bench_growth_monthly)
information_ratio_moderate = fct.ratio_information(tracks_strat_jobs_moderate_with_fees.iloc[1:], bench_balanced_monthly)
information_ratio_def = fct.ratio_information(tracks_strat_jobs_def_with_fees.iloc[1:], bench_defensive_monthly)

st.subheader("Systematic Strategy Portfolio vs Benchmark Performance (with fees applied and Out of Sample)")
fig_sys_strategy_out_of_sample = go.Figure()

fig_sys_strategy_out_of_sample.add_trace(go.Scatter(x=bench_growth_monthly.index, y=bench_growth_monthly, mode='lines', name="Growth"))
fig_sys_strategy_out_of_sample.add_trace(go.Scatter(x=bench_balanced_monthly.index, y=bench_balanced_monthly, mode='lines', name="Balanced"))
fig_sys_strategy_out_of_sample.add_trace(go.Scatter(x=bench_defensive_monthly.index, y=bench_defensive_monthly, mode='lines', name="Defensive"))
fig_sys_strategy_out_of_sample.add_trace(go.Scatter(x=tracks_strat_jobs_growth_with_fees_out_of_sample.index, y=tracks_strat_jobs_growth_with_fees_out_of_sample, mode='lines', name="Growth with Fees (Out of Sample)"))
fig_sys_strategy_out_of_sample.add_trace(go.Scatter(x=tracks_strat_jobs_moderate_with_fees_out_of_sample.index, y=tracks_strat_jobs_moderate_with_fees_out_of_sample, mode='lines', name="Moderate with Fees (Out of Sample)"))
fig_sys_strategy_out_of_sample.add_trace(go.Scatter(x=tracks_strat_jobs_def_with_fees_out_of_sample.index, y=tracks_strat_jobs_def_with_fees_out_of_sample, mode='lines', name="Defensive with Fees (Out of Sample)"))

fig_sys_strategy_out_of_sample.update_layout(
    title="Systematic Strategy Portfolio vs Benchmark Performance (with Fees Applied and Out of Sample)",
    xaxis_title="Time",
    yaxis_title="Portfolio Value",
    yaxis=dict(type="linear"),  # Auto-adapting y-scale
    legend_title="Portfolio"
)

st.plotly_chart(fig_sys_strategy_out_of_sample)

st.write("le ratio d'information de la stratégie growth est de " + "{:.3f}".format(information_ratio_growth))
st.write("le ratio d'information de la stratégie moderate est de " + "{:.3f}".format(information_ratio_moderate))
st.write("le ratio d'information de la stratégie defensive est de " + "{:.3f}".format(information_ratio_def))

# ============
# MARKET VIEWS 
# ============
W1, W2, W3, W4, W5, W6, W7 = mkv.get_market_views()
# Appliquer la fonction de backtest avec ajustement des vues de marché
portfolio_value_with_views = fct.portfolio_with_market_views(tracks,W1,W2,W3,W4,W5,W6,W7)
st.subheader("Portfolio with views vs Benchmark Performance (Partie 6)")
fig_portfolio_views = go.Figure()

fig_portfolio_views.add_trace(go.Scatter(x=bench_growth_monthly.index, y=bench_growth_monthly, mode='lines', name="Growth"))
fig_portfolio_views.add_trace(go.Scatter(x=bench_balanced_monthly.index, y=bench_balanced_monthly, mode='lines', name="Balanced"))
fig_portfolio_views.add_trace(go.Scatter(x=bench_defensive_monthly.index, y=bench_defensive_monthly, mode='lines', name="Defensive"))
fig_portfolio_views.add_trace(go.Scatter(x=portfolio_value_with_views.index, y=portfolio_value_with_views, mode='lines', name="Portfolio including Views"))

fig_portfolio_views.update_layout(
    title="Portfolio with Views vs Benchmark Performance",
    xaxis_title="Time",
    yaxis_title="Portfolio Value",
    yaxis=dict(type="linear"),  # Auto-adapting y-scale
    legend_title="Portfolio"
)

st.plotly_chart(fig_portfolio_views)

# ========================================================================
# BLACK-LITTERMAN BACKTEST FOR RISK PROFILES WITH SPECIFIC DATES AND VIEWS
# ========================================================================
P = mkv.get_viewsBL()           #Market views for BL model
Q = mkv.get_returnsBL()         #Expexted returns for the views

# Dates clés avec vues spécifiques
key_dates_to_views = fct.key_dates_to_views()

# Profils de risque : Défensif, Modéré, Croissance
benchmarks = fct.benchmarks_BL(returns)

#Dictionary to store portfolio values for each profile
portfolio_values = {}

# Streamlit sections for Black-Litterman results
st.subheader("Black-Litterman Results")
names = params.get_names()
# Iterate over risk aversion profiles
for profile, aversion in risk_aversions.items():
    # Execute the Black-Litterman model
    optimal_weights_bl, adjusted_returns_bl = fct.run_black_litterman_model(
        returns, cov_matrix, exp_returns, P, Q, tau, aversion
    )

    # Display optimal weights
    st.write(f"### Optimal Weights for {profile} Profile")
    weights_df = pd.DataFrame({"Asset": names, "Optimal Weight (%)": [weight * 100 for weight in optimal_weights_bl]})
    st.table(weights_df)

    # Display adjusted returns
    st.write(f"### Adjusted Returns for {profile} Profile")
    adjusted_returns_df = pd.DataFrame({"Asset": names, "Adjusted Return (%)": [ret * 100 for ret in adjusted_returns_bl]})
    st.table(adjusted_returns_df)

    # Calculate portfolio values
    portfolio_value_bl = fct.portfolio_with_BL_market_views(
        tracks, returns, cov_matrix, optimal_weights_bl, P, Q, tau, aversion, fct.key_dates_to_views()
    )
    portfolio_values[profile] = portfolio_value_bl
    
# Combine portfolio values into a DataFrame for plotting
portfolio_values_df = pd.DataFrame(portfolio_values)

# Plot portfolio values for each profile
st.write("### Portfolio Values for Different Profiles")
fig_black_litterman = go.Figure()

# Combine portfolio values and benchmarks into a single DataFrame
portfolio_combined_df = pd.concat([portfolio_values_df, pd.DataFrame({
    "Benchmark Growth": bench_growth_monthly,
    "Benchmark Balanced": bench_balanced_monthly,
    "Benchmark Defensive": bench_defensive_monthly
})], axis=1).dropna()

# Add traces for each column
for col in portfolio_combined_df.columns:
    fig_black_litterman.add_trace(go.Scatter(x=portfolio_combined_df.index, y=portfolio_combined_df[col], mode='lines', name=col))

# Update layout
fig_black_litterman.update_layout(
    title="Portfolio Values for Different Profiles (Black-Litterman)",
    xaxis_title="Time",
    yaxis_title="Portfolio Value",
    yaxis=dict(type="linear"),  # Auto-adapting y-scale
    legend_title="Profiles"
)

st.plotly_chart(fig_black_litterman)

    