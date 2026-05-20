@echo off

echo Installing requirements...
pip install -r requirements.txt

echo Starting pipeline...
python -m streamlit run matching_shafar_shasha_version_final.py

pause