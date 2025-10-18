import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Imagen en la barra lateral
st.sidebar.image("images/Logo.png", use_container_width=True)

st.markdown("""
# Modelo de Clasificación XGBoost
## Comparación de Clasificadores XGBoost para Mecanismos de Suicidio

Se implementa un modelo de clasificación **XGBoost** orientado a predecir el mecanismo causal de suicidio en Colombia, utilizando variables sociodemográficas, geográficas y contextuales como predictores. Para evaluar la robustez y rendimiento del algoritmo, se reportan métricas principales de desempeño.
""")

st.markdown("""### Carga del Modelo Entrenado y Evaluación""")

# Carga y preparación de datos
url = "https://raw.githubusercontent.com/jthowinsson/Suicidio_Colombia/main/Presuntos_Suicidios_con_Coor.csv"
data = pd.read_csv(url, encoding="utf-8")

def limpiar_nombres(cols):
    cols = cols.astype(str)
    mapping_tildes = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
    cols = cols.str.translate(mapping_tildes)
    cols = cols.str.lower()
    cols = cols.str.replace(r'[^a-zA-Z0-9]+', '_', regex=True)
    cols = cols.str.strip('_')
    return cols

data.columns = limpiar_nombres(data.columns)
data['mecanismo_causal'] = data['mecanismo_causal'].str.upper().str.strip()
data['target'] = (data['mecanismo_causal'] == 'GENERADORES DE ASFIXIA').astype(int)

categorical_features = ['departamento_del_hecho_dane', 'municipio_del_hecho_dane', 
                        'zona_del_hecho', 'sexo_de_la_victima', 'estado_civil', 
                        'escolaridad', 'manera_de_muerte', 'escenario_del_hecho', 
                        'actividad_durante_el_hecho', 'edad_judicial',
                        'mes_del_hecho', 'dia_del_hecho', 'rango_de_hora_del_hecho_x_3_horas']
numerical_features = ['a_o_del_hecho', 'latitud', 'longitud']
all_features = categorical_features + numerical_features

X = data[all_features]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar el modelo entrenado
modelo = joblib.load("modelo.pkl")

# Realizar predicciones y métricas
y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Mostrar resultados en el panel Streamlit
st.subheader("Resultados del Modelo")
st.metric("Accuracy", f"{accuracy:.4f}")
st.metric("ROC AUC", f"{roc_auc:.4f}")

st.markdown("**Matriz de confusión:**")
st.write(pd.DataFrame(conf_matrix))

st.markdown("**Reporte de clasificación:**")
st.code(class_report)


import plotly.graph_objects as go
from sklearn.metrics import roc_curve
import streamlit as st

st.markdown("### Curva ROC Interactiva del Modelo XGBoost")

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode='lines',
    name=f'Curva ROC (AUC={roc_auc:.3f})',
    line=dict(color='darkblue')
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode='lines',
    name='Línea Base', line=dict(dash='dash', color='gray')
))
fig_roc.update_layout(
    xaxis_title='Tasa de Falsos Positivos (1 - Especificidad)',
    yaxis_title='Tasa de Verdaderos Positivos (Sensibilidad)',
    title='Curva ROC Interactiva - Modelo XGBoost',
    legend=dict(x=0.6, y=0.18),
    width=700, height=460
)

st.plotly_chart(fig_roc, use_container_width=True)


st.markdown("### Visualización gráfica de la matriz de confusión")

import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix

labels = ['Clase 0 = Otros Mecanismos', 'Clase 1 = Generadores de Asfixia']  # Cambia los nombres según tu caso
cm = confusion_matrix(y_test, y_pred)

fig_cm = go.Figure(data=go.Heatmap(
    z=cm, x=labels, y=labels, 
    colorscale='Blues',
    text=cm, texttemplate="%{text}",
    hovertemplate='Real: %{y}<br>Predicción: %{x}<br>N = %{z}<extra></extra>'
))

fig_cm.update_layout(
    xaxis_title="Predicción",
    yaxis_title="Real",
    title="Matriz de Confusión Interactiva",
    width=500, height=430
)

st.plotly_chart(fig_cm, use_container_width=True)





