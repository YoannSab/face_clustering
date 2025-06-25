@echo off
REM Création de l'environnement virtuel
python -m venv venv

REM Activation de l'environnement virtuel
call venv\Scripts\activate.bat

REM Installation des dépendances
pip install -r requirements.txt

REM Lancement du script
python run.py
