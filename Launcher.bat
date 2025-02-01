@echo off
REM Try to find Conda in the PATH
for /f "delims=" %%i in ('where conda 2^>nul') do set CONDA_PATH=%%i

REM Fallback to known Conda locations if not found in PATH
if "%CONDA_PATH%"=="" (
    if exist "C:\Users\%USERNAME%\Anaconda3\condabin\conda.bat" (
        set "CONDA_PATH=C:\Users\%USERNAME%\Anaconda3\condabin\conda.bat"
    ) else if exist "C:\Users\%USERNAME%\Miniconda3\condabin\conda.bat" (
        set "CONDA_PATH=C:\Users\%USERNAME%\Miniconda3\condabin\conda.bat"
    ) else (
        echo "Conda is not installed or not found in a known location. Please install Conda and try again."
        pause
        exit /b
    )
)

REM Activate Conda using the detected or fallback path
call "%CONDA_PATH%" activate

REM Check if the 'streamlit_env' environment exists; create it if not
conda env list | findstr streamlit_env >nul
IF ERRORLEVEL 1 (
    echo "Creating Conda environment 'streamlit_env'..."
    conda create -y -n streamlit_env python=3.9
)

REM Activate the environment
echo "Activating the Conda environment..."
call conda activate streamlit_env

REM Install dependencies
echo "Ensuring dependencies are installed..."
pip install --quiet --upgrade pip
pip install --quiet streamlit pandas plotly altair

REM Navigate to app directory
cd /d "%~dp0"

REM Launch Streamlit app
start "" streamlit run app.py

pause

