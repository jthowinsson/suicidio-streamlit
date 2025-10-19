import streamlit as st

# Imagen en la barra lateral
st.sidebar.image("images/Logo.png", use_container_width=True)

st.header("Datos y metodología")
st.subheader("Fuente de los datos")

st.markdown("""
Se utilizó la base de datos “Presuntos Suicidios en Colombia, 2015–2023 (cifras definitivas)”, disponible en el portal oficial de datos abiertos [6]. Esta fuente contiene 23.544 registros anonimizados, correspondientes a casos de suicidio consumado en el país durante el periodo de estudio.
""")

st.markdown("**Unidad de análisis**")
st.markdown("""
Cada registro representa un caso de suicidio con información individual sobre características sociodemográficas, contexto del hecho y mecanismo causal.
""")

st.markdown("**Población y periodo**")
st.markdown("""
- *Población de estudio*: personas fallecidas por suicidio en Colombia registradas oficialmente.
- *Periodo analizado*: 2015 a 2023.
""")

st.header("Diseño metodológico")

st.markdown("""
1. **Análisis exploratorio de datos (EDA):**
    - Limpieza, verificación de valores faltantes y análisis descriptivo univariado y bivariado.
    - Visualización de tendencias temporales y distribución por sexo, edad, escolaridad y escenario.

2. **Construcción de variable dependiente:**
    - Recodificación de la columna Mecanismo Causal en una variable binaria.

3. **Modelado estadístico:**

Para el análisis de los factores asociados al mecanismo causal, se implementará un modelo de aprendizaje automático avanzado basado en el algoritmo XGBoost (Extreme Gradient Boosting). Específicamente, se utilizará un XGBoost Regressor configurado para una tarea de clasificación logística.

Este modelo se elige por su alta capacidad predictiva y su robustez en el manejo de relaciones complejas y no lineales entre las variables. A diferencia de los modelos de regresión tradicionales, el objetivo del modelo será predecir la probabilidad de que ocurra un mecanismo causal específico (por ejemplo, “Generadores de asfixia" frente a otros) en función de un conjunto de variables predictoras, tales como el sexo, la edad, el estado civil, la escolaridad y la zona del hecho.

De esta manera, se buscará identificar cuáles son los factores con mayor poder de discriminación entre los diferentes mecanismos.

La estructura general de la predicción se puede expresar de la siguiente manera:
""")

st.latex(r"\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)")

st.markdown("""
Donde:

- $\hat{y}_i$ es la predicción bruta (sin transformar) para una observación i.
- $K$ es el número total de árboles de decisión construidos por el modelo.
- $f_k$ representa el k-ésimo árbol de decisión.
- $x_i$ es el conjunto de variables predictoras (edad, sexo, etc.) para la observación i.

Dado que nuestro modelo es de regresión logística (clasificación), esta predicción bruta se transforma en una probabilidad (un valor entre 0 y 1) mediante la función logística o sigmoide:
""")

st.latex(r"P(Y=1 | x_i) = \sigma(\hat{y}_i) = \frac{1}{1 + e^{-\hat{y}_i}}")

st.markdown("""
4. **Métricas de Clasificación**

La validez y capacidad predictiva del modelo se comprobó mediante un riguroso proceso de validación cruzada. Para ello, el conjunto de datos se particionó en un subconjunto de entrenamiento (80%) y otro de prueba (20%). El modelo fue ajustado utilizando el primer grupo, mientras que su desempeño fue evaluado objetivamente sobre el segundo para asegurar que los resultados fueran generalizables.

La eficacia del modelo se cuantificó mediante un conjunto de métricas de clasificación estándar, calculadas a partir de la matriz de confusión. Se emplearon la exactitud (`accuracy`), precisión (`precision`), sensibilidad (`recall`) y la puntuación F1 (`F1-Score`) para valorar el equilibrio entre la correcta identificación de los casos y la minimización de errores de clasificación. Adicionalmente, se utilizó la curva ROC y el área bajo la misma (`AUC`) como medida global de la capacidad discriminativa del modelo para distinguir entre las diferentes categorías de mecanismos.
""")

