import streamlit as st
import pandas as pd
import joblib
from backend import formula_to_vector

# Cargar modelo
model = joblib.load("vrg.pkl")

# Título de la app
st.title("Predicción de Temperatura Crítica de Superconductores")

# Instrucciones
st.markdown("""
Introduce la fórmula química de un superconductor (ej. `Ba1.0La2.0Cu1.0O4.0`) para estimar su temperatura crítica.
""")

# Input del usuario
formula = st.text_input("Fórmula Química", value="Ba1.0La2.0Cu1.0O4.0")

# Al presionar el botón
if st.button("Predecir Temperatura Crítica"):
    try:
        # Transformar la fórmula a vector de características
        vector = formula_to_vector(formula)

        # Realizar la predicción
        prediction = model.predict(vector)[0]

        # Mostrar resultado
        st.success(f"Temperatura crítica estimada: {prediction:.2f} K")
    except Exception as e:
        st.error(f"Ocurrió un error procesando la fórmula: {e}")