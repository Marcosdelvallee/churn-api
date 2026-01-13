from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="API de Predicción de Fuga Bancaria")

# --- 2.BLOQUE DE SEGURIDAD ---
# Esto permite que cualquier origen (*) consuma tu API.
# En producción real, cambiar ["*"] por ["https://tu-dominio.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Cargar el Cerebro y el Scaler al inicio (para que esté listo en RAM)
print("Cargando modelo y scaler...")
model = tf.keras.models.load_model('churn_model.keras')
scaler = joblib.load('scaler.pkl')

# 3. Definir la estructura de datos de entrada (Schema)
class ClienteInput(BaseModel):
    CreditScore: int
    Geography: str  # "France", "Germany", "Spain"
    Gender: str     # "Female", "Male"
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# 4. Crear el Endpoint POST
@app.post("/predict")
def predict_churn(cliente: ClienteInput):
    
    # A. Pre-procesamiento (Convertir el JSON a lo que espera el modelo)
    # Creamos un DataFrame con una sola fila
    input_data = {
        'CreditScore': [cliente.CreditScore],
        'Age': [cliente.Age],
        'Tenure': [cliente.Tenure],
        'Balance': [cliente.Balance],
        'NumOfProducts': [cliente.NumOfProducts],
        'HasCrCard': [cliente.HasCrCard],
        'IsActiveMember': [cliente.IsActiveMember],
        'EstimatedSalary': [cliente.EstimatedSalary],
        # One-Hot Encoding MANUAL (Simulamos lo que hizo get_dummies)
        'Geography_Germany': [1 if cliente.Geography == 'Germany' else 0],
        'Geography_Spain': [1 if cliente.Geography == 'Spain' else 0],
        'Gender_Male': [1 if cliente.Gender == 'Male' else 0]
    }
    
    
    # Forzamos el orden de las columnas:
    columnas_ordenadas = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
        'Geography_Germany', 'Geography_Spain', 'Gender_Male'
    ]
    
    df_input = pd.DataFrame(input_data)
    df_input = df_input[columnas_ordenadas] # Reordenamos por seguridad

    # B. Escalar los datos (Usando el scaler cargado)
    X_scaled = scaler.transform(df_input)

    # C. Predecir
    prediction_prob = model.predict(X_scaled)
    probabilidad = float(prediction_prob[0][0])
    
    se_va = probabilidad > 0.5

    return {
        "probabilidad_fuga": probabilidad,
        "mensaje": "ALERTA: Cliente en riesgo" if se_va else "Cliente seguro",
        "accion_sugerida": "Ofrecer descuento" if se_va else "Nada"
    }