import subprocess
import sys

def global_check():
    checking_streamlit()
    checking_pandas()
    checking_plotly()
    checking_plotlyb()
    checking_scipy()
    checking_openpyxl()
    print('All required modules are installed!')

def checking_streamlit():
    try:
        import streamlit
    except ModuleNotFoundError:
        installing('streamlit')

def checking_pandas():
    try:
        import pandas
    except ModuleNotFoundError:
        installing('pandas')

def checking_plotly():
    try:
        import plotly
    except ModuleNotFoundError:
        installing('plotly')
        
def checking_plotlyb():
    try:
        import matplotlib
    except ModuleNotFoundError:
        installing('matplotlib')        

def checking_scipy():
    try:
        import scipy
    except ModuleNotFoundError:
        installing('scipy')       

def checking_openpyxl():
    try:
        import openpyxl
    except ModuleNotFoundError:
        installing('openpyxl')   

def installing(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    global_check()
