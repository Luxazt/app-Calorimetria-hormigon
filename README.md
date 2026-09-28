
# Predictor de resistencia del hormigón proyectado

Aplicación web del Trabajo de Fin de Grado (UPC). Estima la curva de
resistencia del hormigón proyectado a partir del diseño de la mezcla y de
los resultados de calorimetría, con modelos Random Forest y XGBoost
entrenados por edad de ensayo.

## Ejecución local

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app_2.py
EOF
rm Readme.rtf
