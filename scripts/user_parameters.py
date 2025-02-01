"""=================================================================
 PYTHON FILE FOR USER PARAMETERS: PLEASE MODIFY PARAMETERS ONLY HERE
===================================================================="""
from scripts import function as fct
#SPECIFY YOUR CHOICES IN THE FOLLOWING VARIABLES
#Investors sensibility to market views
tau = 0.5

#Number of day basis in a year.
frequency = 252

#Number of portfolio to randomly generate
n_portfolios = 10000

#Start date and end date for bloomberg data importations
start_date = '20010101'
end_date = '20240830'

#Assets' tickers to import
tickers = ['SLFMQOE FP Equity', 'MSDEWIN Index', 'MSDEEEMN Index', 'LGITRREH Index',
            'LGCPTREH Index', 'LG30TREH Index', 'H04386EU Index']

#Names for columns after data  importation
names = ['Cash', 'Eq DM', 'Eq EM', 'Gov DM', 'Corp DM IG', 'Corp HY', 'Bonds EM AGG']

#Do you want to import data from bloomberg or excel (False for Excel, True for BBG)
user_choice = False

#constraints for maximizing portfolio returns for a given vol
init_weights = [ 0.1 ,  0.2 ,  0.1 ,  0.2 ,  0.2 ,  0.1 ,  0.1 ]
max_weights = [0,  1,  1,  1,  1,  1,  1]
vol_target = 0.04

#Black-Litterman investors risks profiles:
risk_aversions = {
    "Defensive": 7,  # Forte aversion au risque
    "Moderate": 3.5,   # Aversion au risque modérée
    "Growth": 1      # Faible aversion au risque
}


def get_frequency():
    return frequency

def get_n_portfolios():
    return n_portfolios

def get_stend_date():
    return start_date, end_date

def get_tickers():
    return tickers

def get_names():
    return names

def get_user_choice():
    return user_choice

def get_init_weights():
    return init_weights

def get_max_weights():
    return max_weights

def get_target_vol():
    return vol_target

def get_tau():
    return tau

def get_risk_aversions():
    return risk_aversions