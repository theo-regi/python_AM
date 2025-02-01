#Python script for different basic functions:
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# DATA IMPORTATION BASED ON BLOOMBERG
# ============================================================
def blomberg_importation(names, tickers, start_date, end_date):
    import pdblp
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    
    prices = con.bdh(tickers, 'PX_LAST', start_date, end_date)
    prices = prices.dropna()
    prices.columns = names
    prices.to_excel('Prices Frontier.xlsx', index=True)
    prices = pd.read_excel('Prices Frontier.xlsx', parse_dates=['date'], index_col='date')

    return prices

# ============================================================
# DATA IMPORTATION BASED ON GIVEN EXCEL FOR THE PROJECT
# ============================================================
def excel_importation(name):
    prices = pd.read_excel(name, parse_dates=['date'], index_col='date')
    return prices

# ============================================================
# RETURNS AND TRACKS CALCULATION FUNCTIONS
# ============================================================
def calculate_returns(prices):
    return prices.pct_change().dropna()

def calculate_tracks_multidim(returns):
    tracks = (1 + returns).cumprod()
    tracks.iloc[0, :] = 1
    return tracks

def calculate_tracks_unidim(returns):
    tracks = (1 + returns).cumprod()
    tracks.iloc[0] = 1
    return tracks

def plot_tracks(tracks):
    plt.plot(tracks)
    plt.show

# ============================================================
# BASIC INDICATORS AND STATISTICS CALCULATION FUNCTIONS
# ============================================================
def an_returns_from_tracks(tracks, frequency):
    return tracks.tail(1) ** (frequency / len(tracks)) - 1

def an_vol_from_returns(returns, frequency):
    return returns.std() * np.sqrt(frequency)
    
def sharpe_ratio_with_cash(an_returns, an_volatilities):
    return (an_returns - an_returns.iloc[0, 0]) / an_volatilities

def cash_an_returns_from_tracks(tracks, returns, frequency):
    return (1+tracks.tail(1).iloc[0,0]) **(252/len(returns))-1    

def max_DD(track):
    mdd = 0
    peak = track[0]
    for x in track:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd

def cov_matrix(returns, frequency):
    return returns.cov() * frequency


"""Fonction à modifier"""
def expected_returns(an_returns, names, an_volatilities, mdds):
    # exp_returns = an_returns.set_index(names)
    # exp_returns = an_returns.iloc[0,0] + 0.5 * an_volatilities
    exp_returns = an_returns.iloc[0].values
    # exp_returns = an_returns.iloc[0] + (np.sqrt(an_volatilities) - mdds) / 12
    return exp_returns

"""Non appellée pour le moment car inutile dans le reste du code"""
def expected_sharpes(exp_returns, an_returns, an_volatilities):
    exp_sharpes = (exp_returns - an_returns.iloc[0]) / an_volatilities
    return exp_sharpes

# =================================================================
# PORTFOLIOS GENERATOR FOR OPTIMAL FRONTIER AND CAPITAL MARKET LINE
# =================================================================
def generate_portfolios(n_portfolios, returns, exp_returns, an_returns, an_volatilities, frequency):
    cov = returns.cov() * frequency
    mean_variance_pairs = []
    random_weights = []

    for k in range(n_portfolios):
        # Generate a random portfolio
        weights = np.random.rand(len(exp_returns))
        weights = weights / sum(weights)
        random_weights.append(weights)
        
        # Increment portfolio characteristics
        portfolio_E_Return = np.dot(weights, exp_returns)
        portfolio_E_Variance = np.dot(weights, np.dot(cov, weights))
        mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])

    mean_variance_pairs = np.array(mean_variance_pairs)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=mean_variance_pairs[:,1]**0.5,
        y=mean_variance_pairs[:,0],
        marker=dict(color=(mean_variance_pairs[:,0]-exp_returns[0])/(mean_variance_pairs[:,1]**0.5),
                    showscale=True,
                    size=7,
                    line=dict(width=1),
                    colorscale="RdBu",
                    colorbar=dict(title="Sharpe<br>Ratio")),
        mode='markers'))

    fig.add_trace(go.Scatter(
         x=np.array(an_volatilities),
         y=np.array(an_returns),
         mode='markers',
         marker=dict(color='green', size=100), name='Assets'))

    fig.update_layout(
        template='plotly_white',
        xaxis=dict(title='Annualised Risk (Volatility)'),
        yaxis=dict(title='Annualised Return'),
        title='Sample of Random Portfolios',
        coloraxis_colorbar=dict(title="Sharpe Ratio"))

    #pio.renderers.default='browser'
    #fig.show()
    return weights, mean_variance_pairs, fig

def capital_market_line_constructor(mean_variance_pairs, risk_free_rate):
    # Calculate Sharpe ratios for all random portfolios on the efficient frontier
    sharpe_ratios = (mean_variance_pairs[:, 0] - risk_free_rate) / (mean_variance_pairs[:, 1]**0.5)

    # Find the portfolio with the maximum Sharpe ratio (tangency portfolio)
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_return = mean_variance_pairs[max_sharpe_idx, 0]
    max_sharpe_volatility = mean_variance_pairs[max_sharpe_idx, 1]**0.5

    # Capital Market Line (CML) - a straight line from the risk-free rate through the tangency portfolio
    # We extend the CML to a higher level of volatility (e.g., up to 1.5x the maximum volatility)
    cml_volatilities = np.linspace(0, max_sharpe_volatility * 1.5, 100)
    cml_returns = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_volatility * cml_volatilities

    # Plot the efficient frontier and the Capital Market Line
    fig = go.Figure()

    # Plot efficient frontier (random portfolios)
    fig.add_trace(go.Scatter(
        x=mean_variance_pairs[:, 1]**0.5,  # Volatility (standard deviation)
        y=mean_variance_pairs[:, 0],       # Return
        mode='markers',
        marker=dict(
            color=sharpe_ratios,           # Color based on Sharpe ratio
            showscale=True,
            size=7,
            line=dict(width=1),
            colorscale="RdBu",
            colorbar=dict(title="Sharpe<br>Ratio")
        ),
        name='Efficient Frontier'
    ))

    # Plot the Capital Market Line (CML)
    fig.add_trace(go.Scatter(
        x=cml_volatilities,
        y=cml_returns,
        mode='lines',
        name='Capital Market Line (CML)',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Customize layout
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(title='Annualized Risk (Volatility)'),
        yaxis=dict(title='Annualized Return'),
        title='Efficient Frontier and Capital Market Line',
        showlegend=True
    )

    # Show the plot
    #pio.renderers.default='browser'
    #fig.show()
    
    return max_sharpe_return, max_sharpe_volatility, fig

# Function for the CML equation
def cml_return(risk_free_rate, max_sharpe_return, max_sharpe_volatility, volatility):
    return risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_volatility * volatility

# =============================================================================
# FUNCTIONS FOR MAXIMIZATION OF PORTFOLIO RETURNS FOR A GIVEN VOLATILITY TARGET
# =============================================================================
def ptfReturn(returns, weights, frequency):
    tracks = (1 + returns).cumprod()
    tracks.iloc[0,:] = 1
    an_returns = tracks.tail(1) ** (frequency / len(returns)) - 1
    ptf_return = np.dot(an_returns, weights)
    return ptf_return

def ptfVol(returns, weights, frequency):
    cov = returns.cov() * frequency
    ptf_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    return ptf_vol

def constraint_weights(weights):
    # Weights sum to 1
    return {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

def constraint_weights_bounds(weights, init_weights):
    # Weights between 0 & 1
    constraint_weights_bounds = tuple((0, 1) for weights in init_weights)
    return constraint_weights_bounds

def constraint(weights, max_weights):
    return max_weights - weights

def constraint_weights_by_asset(weights, max_weights):
    return {'type': 'ineq', 'fun': lambda weights: max_weights - weights}
    
# Minimization of portfolio volatility for a given return target
def minVol(returns, return_target, frequency, constraint_weight_by_asset, init_weights):
    # Usual constraints and close to return target
    constraint_return = {'type': 'eq', 'fun': lambda weights: ptfReturn(returns, weights, frequency) - return_target}
    constraints = [constraint_return, constraint_weights, constraint_weight_by_asset]
    result = minimize(ptfVol, init_weights, method='SLSQP', constraints=constraints, bounds=constraint_weights_bounds)
    return result.x

# Maximization of portfolio return for a given volatility target
def maxReturn(returns, vol_target, frequency, weights, constraint_weights, constraint_weight_by_asset, constraint_weights_bound, init_weights):
    def ptfNegReturn(weights):
        return -ptfReturn(returns, weights, frequency)

    # Usual constraints and close to vol target
    constraint_vol = {'type': 'eq', 'fun': lambda weights: ptfVol(returns, weights, frequency) - vol_target}
    constraints = [constraint_vol, constraint_weights, constraint_weight_by_asset]
    result = minimize(ptfNegReturn, init_weights, method='SLSQP', constraints=constraints, bounds=constraint_weights_bound)
    return result.x

def maxSharpe(returns, frequency, an_returns, weights, constraint_weights, constraint_weight_by_asset, constraint_weights_bound, init_weights):
    # Calculate the expected returns and covariance matrix
    cov = returns.cov() * frequency
    exp_returns = an_returns.iloc[0].values  # Expected returns
    risk_free_rate = exp_returns[0]  # Assuming the first asset is risk-free (Cash)

    def negSharpe(weights):
        # Portfolio return and volatility
        ptf_return = np.dot(weights, exp_returns)
        ptf_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        
        # Sharpe ratio (we negate it to minimize the negative Sharpe ratio)
        sharpe_ratio = (ptf_return - risk_free_rate) / ptf_vol
        return -sharpe_ratio

    # Constraints: Weights sum to 1, each weight between 0 and 1, and asset-specific constraints
    constraints = [constraint_weights, constraint_weight_by_asset]
    result = minimize(negSharpe, init_weights, method='SLSQP', constraints=constraints, bounds=constraint_weights_bound)

    return result.x

# ===============================================
# FUNCTIONS FOR BENCHMARK PORTFOLIOS CONSTRUCTION
# ===============================================
def benchmark_portfolio(weight_cash, weight_ptf, returns_cash, returns_ptf):
    benchmark_returns = weight_cash * returns_cash + weight_ptf * returns_ptf
    bench_perf = (1+benchmark_returns).cumprod()
    bench_perf[0] = 1
    return bench_perf

def plot_benchmarks(bench_defensive, bench_balanced, bench_growth):
    plt.figure(figsize=(10, 8))
    plt.plot(bench_defensive, label='portefeuille defensif', color='red', linestyle='-')
    plt.plot(bench_balanced, label='portefeuille moderate', color='green', linestyle='-')
    plt.plot(bench_growth, label='portefeuille growth', color='blue', linestyle='-')
    #plt.legend(loc='upper left')  # Positionne la légende en haut à gauche
    plt.show()

def plot_one_ptf(tracks_ptf, name_ptf, title):
    plt.figure(figsize=(10, 8))
    plt.plot(tracks_ptf, label=name_ptf, color='purple', linestyle='-')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend(loc='upper left')
    plt.show()

# ================================
# FUNCTIONS TO REBALANCE PORFOLIOS
# ================================
def rebalance_portfolio(weights, returns):
    portfolio_value = [1]  # Start with portfolio value of 1
    
    for i in range(1, len(returns)):
        # Calculate portfolio return for the current month
        portfolio = (1 + (returns.iloc[i] * weights).sum())
        new_value = portfolio_value[-1] * portfolio
        portfolio_value.append(new_value)
        # Rebalance to target weights at the end of the month
        weights = weights / sum(weights)  # Ensure weights sum to 1 after rebalancing
    
    return pd.Series(portfolio_value, index=returns.index)

# =====================================
# FUNCTIONS FOR THE SYSTEMATIC STRATEGY
# =====================================
def prepare_us_jobs(data_macro):
    variation_points = data_macro[data_macro['US Jobs'].diff().ne(0)].index
    monthly_start = data_macro.index.to_period('M').to_timestamp()
    monthly_starts = data_macro[data_macro.index.isin(monthly_start)].index
    important_dates = sorted(monthly_starts.union(variation_points))
    return data_macro.loc[important_dates, 'US Jobs'].tolist()

def strategy_indicator(us_jobs, coefficient_indicator):
    results = [0.00] * 10
    results50 = [0.00] * 30
    results_moy = []
    results_moy50 = []
    increase_eq = []
    
    for i in range(len(us_jobs)):
        results.pop(0)
        results50.pop(0)
        results.append(us_jobs[i])
        results50.append(us_jobs[i])

        '''results_E3M.pop(0)
        results_E3M.append(filtered_E3M[i])
        moyenne_E3M = np.mean(results_E3M)
        std_E3M = np.std(results_E3M)
        
        if filtered_E3M[i]<moyenne_E3M +2* std_E3M:
            high_E3M.append(False)
        else:
            high_E3M.append(True)'''
        
        mean20 = np.mean(results)
        mean50 = np.mean(results50)
        results_moy.append(mean20)
        results_moy50.append(mean50)
        if mean20<0:
            if mean20 < mean50:
                increase_eq.append(False)
            else:
                increase_eq.append(True)
        else:
            if mean20*coefficient_indicator < mean50:
                increase_eq.append(False)
            else:
                increase_eq.append(True)
    
    #plt.plot(important_dates,results_moy)
    #plt.plot(important_dates,results_moy50)
    #plt.show()
    
    return increase_eq

#faire une double sortie qui sortira toutes les dates pour lesquelles il y a eu un changement et le montant du changement
def rebalanced_returns(tracks, weights_maxsharpe, increase_eq,weights_exposed_equity,weights_exposed_oblig_E3M_high,coef):
    tracks_end_of_month = tracks.resample('M').last()
    tracks_start_of_month = tracks.resample('M').first()
    
    monthly_returns = (tracks_end_of_month / tracks_start_of_month) - 1
    
    equity = True
    oblig = True
    dates_transaction =[]
    transaction_fees =[]
    
    for i in range(0, len(increase_eq)):#len(increase_eq)
        if increase_eq[i] == True:
            new_weights = weights_exposed_equity
            if oblig==True:
                weights_change = np.abs((np.array(new_weights) - weights_maxsharpe)).sum()
                weights_maxsharpe = new_weights
                oblig = False
                equity = True
                transaction_fees.append(weights_change*0.0005)
                dates_transaction.append(monthly_returns.index[i])
        elif increase_eq[i] == False:
            new_weights = weights_exposed_oblig_E3M_high
            if equity == True:
                weights_change = np.abs(np.array(new_weights) - weights_maxsharpe).sum()
                weights_maxsharpe = new_weights
                oblig = True
                equity = False
                transaction_fees.append(weights_change*0.0005)
                dates_transaction.append(monthly_returns.index[i])
        monthly_returns.iloc[i] = monthly_returns.iloc[i] * new_weights
        df_transaction = pd.DataFrame({'Dates transaction' : dates_transaction,
                          'Frais transaction' : transaction_fees})
    monthly_returns.iloc[-1] = monthly_returns.iloc[-1] * new_weights
    return monthly_returns,df_transaction

def plot_strategies_vs_benchmarks(tracks_strat_jobs_growth, tracks_strat_jobs_moderate, tracks_strat_jobs_def,
                                  ptf_def_perf, ptf_growth_perf, ptf_bal_perf, with_fees):
    plt.figure(figsize=(10, 8))
    plt.plot(tracks_strat_jobs_growth)
    plt.plot(tracks_strat_jobs_moderate)
    plt.plot(tracks_strat_jobs_def)
    plt.plot(ptf_def_perf)
    plt.plot(ptf_growth_perf)
    plt.plot(ptf_bal_perf)
    if with_fees == False:
        l = ['strat_jobs qui bat le growth','strat_jobs qui bat le moderate','strat_jobs qui bat le defensif', 
         'defensive', 'growth', 'balanced']
        plt.title('Comparaison des Stratégies Optimisées et des Benchs')
    else:
        l = ['Stratégie Growth avec Frais','Stratégie Moderate avec Frais','Stratégie Défensive avec Frais', 
             'defensive', 'growth', 'balanced']
        plt.title('Comparaison des Stratégies Optimisées et des Benchmarks avec Frais')
        
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur du Portefeuille', fontsize=12)
    plt.legend(l)
    plt.show()
    
def apply_management_fees(portfolio_perf, annual_fee,transaction_fees):
    
    mensual_fee = annual_fee/12
    # Créer une copie de la performance pour ne pas modifier l'original
    portfolio_with_fees = portfolio_perf.copy()
    k=0
    # Appliquer les frais à la fin de chaque mois
    for i in range(len(portfolio_with_fees)):
        portfolio_with_fees[i] -= mensual_fee
        if portfolio_with_fees.index[i] == transaction_fees.iloc[k,0]:
            portfolio_with_fees[i] -= transaction_fees.iloc[k,1]
            if k <8:
                k=k+1
            else:
                k=8
    return (1+portfolio_with_fees).cumprod()

# =====================================
# FUNCTIONS FOR OUT OF SAMPLE
# =====================================

def rebalanced_returns_out_of_sample(tracks, weights_maxsharpe, increase_eq,weights_exposed_equity,weights_exposed_oblig_E3M_high,coef):
    tracks_end_of_month = tracks.resample('M').last()
    tracks_start_of_month = tracks.resample('M').first()
    
    monthly_returns = (tracks_end_of_month / tracks_start_of_month) - 1
    
    equity = True
    oblig = True
    dates_transaction =[]
    transaction_fees =[]
    
    for i in range(0, len(increase_eq)-1):#len(increase_eq)
        if increase_eq[i] == True:
            new_weights = weights_exposed_equity
            if oblig==True:
                weights_change = np.abs((np.array(new_weights) - weights_maxsharpe)).sum()
                weights_maxsharpe = new_weights
                oblig = False
                equity = True
                transaction_fees.append(weights_change*0.0005)
                dates_transaction.append(monthly_returns.index[i])
        elif increase_eq[i] == False:
            new_weights = weights_exposed_oblig_E3M_high
            if equity == True:
                weights_change = np.abs(np.array(new_weights) - weights_maxsharpe).sum()
                weights_maxsharpe = new_weights
                oblig = True
                equity = False
                transaction_fees.append(weights_change*0.0005)
                dates_transaction.append(monthly_returns.index[i])
        monthly_returns.iloc[i+1] = monthly_returns.iloc[i+1] * new_weights
        df_transaction = pd.DataFrame({'Dates transaction' : dates_transaction,
                          'Frais transaction' : transaction_fees})
    monthly_returns.iloc[-1] = monthly_returns.iloc[-1] * new_weights
    return monthly_returns,df_transaction


#Calcul du ration d'information
def ratio_information(ptf_return, bench_return):
    # Align the two series by index
    aligned_ptf_return, aligned_bench_return = ptf_return.align(bench_return, join='inner')
    
    # Calculate excess return
    excess_return = aligned_ptf_return - aligned_bench_return
    
    # Calculate mean and standard deviation of excess return
    mean_excess_return = np.mean(excess_return)
    std_excess_return = np.std(excess_return)
    
    # Information ratio
    information_ratio = mean_excess_return / std_excess_return
    
    return information_ratio



# =====================================
# FUNCTIONS FOR MARKET VIEWS
# =====================================
# Fonction pour ajuster les poids en fonction des vues de marché
def adjust_weights(date, W1, W2, W3, W4, W5, W6, W7):
    if date <= pd.Timestamp('2007-06-30'):
        return np.array(W1)
    elif date <= pd.Timestamp('2009-03-09'):
        return np.array(W2)
    elif date <= pd.Timestamp('2019-12-31'):
        return np.array(W3)
    elif date <= pd.Timestamp('2020-03-15'):
        return np.array(W4)
    elif date <= pd.Timestamp('2021-12-31'):
        return np.array(W5)
    elif date <= pd.Timestamp('2022-12-31'):
        return np.array(W6)
    else:
        return np.array(W7)

# Fonction de backtesting avec ajustement des vues de marché
def portfolio_with_market_views(tracks,W1,W2,W3,W4,W5,W6,W7):
    portfolio_value = [1]  # Valeur initiale du portefeuille (base 1)

    # Boucle à travers les dates journalières
    for i in range(1, len(tracks)):
        # Ajuster les poids en fonction des dates de vues de marché
        current_date = tracks.index[i]
        weights = adjust_weights(current_date, W1,W2,W3,W4,W5,W6,W7)

        # Calcul des rendements journaliers du portefeuille
        daily_returns = (tracks.iloc[i] / tracks.iloc[i - 1]) - 1

        # Calcul du rendement du portefeuille pondéré par les actifs
        portfolio_daily_return = (daily_returns * weights).sum()

        # Calcul de la nouvelle valeur du portefeuille (cumulatif)
        new_portfolio_value = portfolio_value[-1] * (1 + portfolio_daily_return)

        # Mise à jour de la valeur du portefeuille
        portfolio_value.append(new_portfolio_value)

    # Retourner la série du portefeuille avec les bonnes dates comme index
    return pd.Series(portfolio_value, index=tracks.index)

def plot_strategies_vs_benchmarks_market_view(portfolio_value_with_views,ptf_def_perf, ptf_growth_perf, ptf_bal_perf):
    plt.figure(figsize=(10, 8))
    plt.plot(portfolio_value_with_views)
    plt.plot(ptf_def_perf)
    plt.plot(ptf_growth_perf)
    plt.plot(ptf_bal_perf)
    l = ['Stratégie Market View','strat_jobs qui bat le moderate','strat_jobs qui bat le defensif', 
     'defensive', 'growth', 'balanced']
    plt.title('Comparaison de la stratégie market view et des Benchs')
        
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur du Portefeuille', fontsize=12)
    plt.legend(l)
    plt.show()


# =====================================
# BLACK-LITTERMAN MODEL FUNCTIONS
# =====================================
# Fonction pour obtenir les rendements en équilibre
# Rendements basés sur la moyenne historique des rendements observés des différents actifs
def equilibrium_returns(cov_matrix, market_weights, risk_aversion):
    return risk_aversion * np.dot(cov_matrix, market_weights)

# Fonction pour combiner les rendements en équilibre avec les vues de marché
# Les vues de marché ajustent les rendements prédictifs basés sur les prévisions des investisseurs
def black_litterman(equilibrium_returns, cov_matrix, P, Q, omega, tau):
    if P.ndim == 1:
        P = P.reshape(1, -1)
    inv_cov_matrix = np.linalg.inv(tau * cov_matrix)
    part1 = np.linalg.inv(inv_cov_matrix + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))
    part2 = np.dot(inv_cov_matrix, equilibrium_returns) + np.dot(np.dot(P.T, np.linalg.inv(omega)), Q)
    posterior_returns = np.dot(part1, part2)
    return posterior_returns

# Fonction pour maximiser l'utilité avec une aversion au risque donnée (Markowitz ajusté)
def markowitz_with_risk_aversion(posterior_returns, cov_matrix, risk_aversion):
    num_assets = len(posterior_returns)

    def utility(weights):
        expected_return = np.dot(weights, posterior_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return - (expected_return - (risk_aversion / 2) * portfolio_variance)

    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'eq', 'fun': lambda weights: weights[0]-0}]
    bounds = [(0, 1) for _ in range(num_assets)]
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(utility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

# Fonction pour exécuter le modèle Black-Litterman et obtenir les poids optimaux et les rendements postérieurs
def run_black_litterman_model(returns, cov_matrix, market_weights, P, Q, tau, risk_aversion):
    # Calcul des rendements en équilibre
    equilibrium_ret = equilibrium_returns(cov_matrix, market_weights, risk_aversion)
    
    # Calcul des rendements postérieurs (Black-Litterman)
    omega = np.diag(np.full(len(Q), 0.02))  # Définir la matrice Omega (incertitude)
    posterior_returns = black_litterman(equilibrium_ret, cov_matrix, P, Q, omega, tau)
    
    # Optimisation des poids
    optimal_weights = markowitz_with_risk_aversion(posterior_returns, cov_matrix, risk_aversion)

    return optimal_weights, posterior_returns


# Fonction pour ajuster les poids selon les vues de marché spécifiques et la date
def adjust_weights_litterman(current_date, returns, cov_matrix, market_weights, P, Q, tau, aversion, key_dates_to_views):
    """
    Ajuste les poids en fonction des vues de marché spécifiques associées aux dates définies.
    """
    if current_date in key_dates_to_views:
        # Récupérer les indices des vues à utiliser pour la date actuelle
        views_indices = key_dates_to_views[current_date]
        
        # Filtrer les vues et les rendements espérés en fonction des indices
        P_filtered = P[views_indices]
        Q_filtered = Q[views_indices]
        
        # Calcul des poids optimaux en utilisant les vues filtrées
        optimal_weights, _ = run_black_litterman_model(returns, cov_matrix, market_weights, P_filtered, Q_filtered, tau, aversion)
        print(current_date, optimal_weights)
        for i,  optiweights in enumerate( optimal_weights):
            print(f" {i+1}: {optiweights:.2%}")
        return optimal_weights
        
    else:
        return market_weights

# Fonction de backtesting avec ajustement des vues de marché (Black-Litterman)
def portfolio_with_BL_market_views(tracks, returns, cov_matrix, market_weights, P, Q, tau, aversion, key_dates_to_views):
    portfolio_value = [1]  # Valeur initiale du portefeuille (base 1)

    for i in range(1, len(tracks)):
        current_date = tracks.index[i]
        
        # Ajustement des poids avec Black-Litterman en fonction des vues sélectionnées
        weights = adjust_weights_litterman(current_date, returns, cov_matrix, market_weights, P, Q, tau, aversion, key_dates_to_views)
        market_weights=weights
        # Vérifier la longueur des poids par rapport au nombre d'actifs
        if len(weights) != len(tracks.columns):
            raise ValueError(f"Les poids optimaux ({len(weights)}) ne correspondent pas au nombre d'actifs ({len(tracks.columns)}).")

        # Calcul des rendements journaliers
        daily_returns = (tracks.iloc[i] / tracks.iloc[i - 1]) - 1

        # Vérifiez également que la longueur de `daily_returns` est la même que `weights`
        if len(daily_returns) != len(weights):
            raise ValueError(f"La longueur des rendements journaliers ({len(daily_returns)}) ne correspond pas à celle des poids ({len(weights)}).")
        
        # Calcul du rendement du portefeuille pondéré par les actifs
        portfolio_daily_return = (daily_returns * weights).sum()
        new_portfolio_value = portfolio_value[-1] * (1 + portfolio_daily_return)

        portfolio_value.append(new_portfolio_value)

    return pd.Series(portfolio_value, index=tracks.index)


def key_dates_to_views():
    return {
        pd.Timestamp('2002-08-02'): [3,5, 13],                # À cette date, les vues 4, 6 et 14 s'appliquent
        pd.Timestamp('2007-06-29'): [1,7, 8, 10, 16],         # À cette date, les vues 2, 8, 9, 11 et 16 s'appliquent
        pd.Timestamp('2009-03-09'): [2, 12,14,15],            # À cette date, les vues 3, 13, 15 et 16 s'appliquent
        pd.Timestamp('2019-12-31'): [1, 7, 8, 9, 10, 16],     # À cette date, les vues 8 et 9 s'appliquent
        pd.Timestamp('2020-03-16'): [2,3,4,5,6,12,15],        # À cette date, les vues 10 et 11 s'appliquent
        pd.Timestamp('2021-12-31'): [1,2, 6, 7,8, 9, 15, 16], # À cette date, les vues 12 et 13 s'appliquent
        pd.Timestamp('2022-12-30'): [2, 4, 12, 14, 15],       # À cette date, les vues 14, 15 et 16 s'appliquent
        pd.Timestamp('2024-11-04'):[2, 12, 14],               # 100% sur le développés apres victoire de trump, on peut rajouter une vue ici si on veut
        pd.Timestamp('2025-04-17'): [1, 5, 7, 9, 16, 18]      #potentiel de forte inflation après l'arrivée de Trump au pouvoir et donc de hausse des taux directeur de la FED --> full obligations
    }

def benchmarks_BL(returns):
    benchmarks = {
        "Defensive": benchmark_portfolio(0.5, 0.5, returns['Cash'], returns.mean(axis=1)),
        "Moderate": benchmark_portfolio(0.25, 0.75, returns['Cash'], returns.mean(axis=1)),
        "Growth": benchmark_portfolio(0, 1, returns['Cash'], returns.mean(axis=1))
    }
    
    return benchmarks















