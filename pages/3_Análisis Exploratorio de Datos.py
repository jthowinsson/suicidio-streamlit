import streamlit as st

# Imagen en la barra lateral
st.sidebar.image("images/Logo.png", use_container_width=True)

import streamlit as st

st.header("Análisis Exploratorio de Datos")

st.markdown("""
El Análisis Exploratorio de Datos (EDA) constituye una etapa fundamental en toda investigación estadística, ya que permite comprender las características principales del conjunto de datos, identificar patrones, detectar posibles inconsistencias y orientar la preparación de la información antes del modelado.

En este estudio, el EDA tiene como propósito examinar los registros de suicidios en Colombia durante el periodo 2015–2023, con el fin de describir las variables sociodemográficas y contextuales asociadas y evaluar la calidad de los datos para la construcción de un modelo de regresión XGBoost Regressor.

En primer lugar, se realiza una revisión general del dataset, verificando el número de observaciones, la disponibilidad de variables y la presencia de valores faltantes o categorías indeterminadas. Posteriormente, se lleva a cabo un análisis univariado, en el que se describen la distribución de la variable dependiente (mecanismo de suicidio: generadores de asfixia vs otros) y de las principales variables independientes (sexo, edad, estado civil, escolaridad, escenario, zona, entre otras).

En una segunda fase, se desarrolla un análisis bivariado, explorando las asociaciones preliminares entre el mecanismo de suicidio y las variables sociodemográficas/contextuales. Esto incluye el uso de tablas de contingencia, gráficos comparativos y estimaciones de razones de momios (odds ratios) crudas en variables binarias.

Finalmente, a partir de los hallazgos del EDA, se definen las decisiones de recodificación, agrupación y preparación de variables que alimentarán el modelo de regresión XGBoost Regressor, asegurando la validez y consistencia de los resultados.
""")

import streamlit as st

st.header("1. Análisis Univariado")
st.subheader("Desarrollo del Análisis Exploratorio de Datos")
st.markdown("**Carga, estandarización y dimensiones del DataFrame:**")

st.markdown("""
El dataFrame contiene 23,137 registros y 42 variables, lo que proporciona una base sólida para el análisis estadístico y la modelización, permitiendo explorar diversas características sociodemográficas y contextuales relacionadas con los suicidios en Colombia. Fuente: [Datos Abiertos](https://www.datos.gov.co/Justicia-y-Derecho/Presuntos-Suicidios-Colombia-2015-a-2024-Cifras-de/f75u-mirk/about_data), la cual es actualizada periódicamente por el Ministerio de Salud y Protección Social de Colombia.

Las variables incluyen información sobre el año del evento, la edad, el sexo, el estado civil, la escolaridad, el escenario del suicidio, la zona (urbana/rural), el departamento y municipio de ocurrencia, entre otras. Esta diversidad de variables permite un análisis integral de los factores asociados a los suicidios en el país.

Adicionalmente, se observa que algunas variables presentan valores faltantes o categorías indeterminadas, lo que requerirá una atención especial durante el análisis para asegurar la calidad y validez de los resultados.

Y finalmente al DataFrame se le adicionan dos columnas nuevas: Longitud y Latitud, las cuales son obtenidas a partir de la columna Municipio de ocurrencia, mediante la librería geopy. Fuente: [DANE](https://geoportal.dane.gov.co/servicios/descarga-y-metadatos/datos-geoestadisticos/).
""")



import streamlit as st
import pandas as pd
import plotly.express as px

st.subheader("Mapeo, Estandarización de las Variables y dimensiones del DataFrame")

st.markdown("**1. Estandarización de Nombres de Columnas:**")

# Cargar los datos
url = "https://raw.githubusercontent.com/jthowinsson/Suicidio_Colombia/main/Presuntos_Suicidios_con_Coor.csv"
df = pd.read_csv(url, encoding="utf-8")

# Guardar nombres originales
nombres_originales = df.columns.tolist()

# Función para limpiar nombres de columnas
def limpiar_nombres(cols):
    cols = pd.Series(cols, dtype="str")
    # Quitar tildes
    mapping_tildes = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
    cols = cols.str.translate(mapping_tildes)
    # A minúsculas
    cols = cols.str.lower()
    # Reemplazar cualquier carácter no alfanumérico por guion bajo
    cols = cols.str.replace(r'[^a-zA-Z0-9]+', '_', regex=True)
    # Limpiar guiones bajos extra al inicio/final
    cols = cols.str.strip('_')
    return cols

# Aplicar limpieza
df.columns = limpiar_nombres(df.columns)
nombres_limpios = df.columns.tolist()

# Tabla comparativa
tabla = pd.DataFrame({
    'Nombre original': nombres_originales,
    'Nombre estandarizado': nombres_limpios
})

st.dataframe(tabla, use_container_width=True)

st.markdown("""
<em>Fuente: <a href="https://www.datos.gov.co/Justicia-y-Derecho/Presuntos-Suicidios-Colombia-2015-a-2024-Cifras-de/f75u-mirk/about_data" target="_blank">datos.gov.co</a></em>
""", unsafe_allow_html=True)

# Mostrar dimensiones en Streamlit
st.info(f"Dimensiones del DataFrame: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")

st.markdown("""
Se hace necesario estandarizar las variables categóricas para asegurar la consistencia en el análisis univariado y bivariado. Esto incluye:

- Convertir todas las entradas a minúsculas para evitar duplicados por diferencias de mayúsculas.
- Eliminar espacios en blanco al inicio y final de las cadenas.
- Reemplazar valores faltantes o ambiguos (como 'nan', 'n/a', 'desconocido') por una categoría uniforme, por ejemplo 'Falta_dato'.
""")

st.markdown("### 2. Información general del DataFrame")

import streamlit as st
import pandas as pd
import io

import streamlit as st
import pandas as pd

# Crear tabla resumen: nombre, tipo y nulos
tabla_info = pd.DataFrame({
    'Columna': df.columns,
    'Tipo de dato': df.dtypes.astype(str),
    'Valores nulos': df.isnull().sum()
})

st.dataframe(tabla_info, use_container_width=True)

# Resaltado académico
st.markdown("""
<em>Tabla: Información general de columnas, tipo de dato y valores nulos del dataset.</em>
""", unsafe_allow_html=True)

st.markdown("### 3. Visualizar las primeras filas del DataFrame")

st.dataframe(df.head(10), use_container_width=True)

st.markdown("""
<em>Nota: Solo se muestran las primeras 10 filas. Puedes ajustar el número según el análisis o dejar que el usuario defina cuántas ver.</em>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd

st.markdown("### 4. Resumen de tipos de variables y valores faltantes")

# Tabla 1: Dimensiones
dimensiones = pd.DataFrame({
    "Registros": [df.shape[0]],
    "Variables": [df.shape[1]]
})

# Tabla 2: Conteo por tipo de variable
n_numeric   = df.select_dtypes(include=["number"]).shape[1]
n_categ     = df.select_dtypes(include=["object", "category"]).shape[1]
n_logical   = df.select_dtypes(include=["bool"]).shape[1]
n_datetime  = df.select_dtypes(include=["datetime64"]).shape[1]

tipos_tbl = pd.DataFrame({
    "Tipo de variable": ["Numéricas", "Categóricas", "Lógicas", "Fecha / Fecha-hora"],
    "Conteo": [n_numeric, n_categ, n_logical, n_datetime]
})
tipos_tbl["% sobre total"] = (100 * tipos_tbl["Conteo"] / df.shape[1]).round(1)

# Tabla 3: Valores faltantes por variable
faltantes_tbl = pd.DataFrame({
    "Columna": df.columns,
    "Valores faltantes": df.isnull().sum(),
    "% Faltante": (100 * df.isnull().sum() / df.shape[0]).round(1)
})

# Mostrar tablas
st.markdown("**Dimensiones del dataset:**")
st.dataframe(dimensiones)

st.markdown("**Variables por tipo:**")
st.dataframe(tipos_tbl, use_container_width=True)

st.markdown("**Valores faltantes por variable:**")
st.dataframe(faltantes_tbl.sort_values("Valores faltantes", ascending=False), use_container_width=True)

st.markdown("""
<em>Nota: Las variables 'Categóricas' incluyen 'object' y 'category'; 'Fecha' incluye 'datetime64' (con o sin zona horaria).</em>
""", unsafe_allow_html=True)

st.markdown("### 5. Número de valores únicos por columna categórica")

# Seleccionar columnas categóricas y contar valores únicos
unicos = df.select_dtypes(['object', 'category']).nunique().reset_index()
unicos.columns = ['Columna categórica', 'Valores únicos']

# Mostrar tabla ordenada de mayor a menor
st.dataframe(unicos.sort_values('Valores únicos', ascending=False), use_container_width=True)

st.markdown("""
<em>Nota: Esta tabla ayuda a identificar posibles variables para recodificación, agrupamiento o análisis de outliers en variables categóricas.</em>
""", unsafe_allow_html=True)


st.markdown("### 6. Estadísticas descriptivas de variables categóricas")

# Resumen de variables categóricas
desc_categoricas = df.describe(include='object').T.reset_index()
desc_categoricas = desc_categoricas.rename(columns={'index':'Columna', 'unique':'Valores únicos', 'top':'Valor más frecuente', 'freq':'Frecuencia'})

st.dataframe(desc_categoricas, use_container_width=True)

st.markdown("""
<em>Nota: Para cada variable categórica, se resumen las categorías únicas, el valor más común y su frecuencia. Útil para el diagnóstico de la distribución de clases y potenciales recodificaciones.</em>
""", unsafe_allow_html=True)

st.markdown("### 7. Valores nulos por columna")

nulos_tbl = pd.DataFrame({
    'Columna': df.columns,
    'Valores nulos': df.isnull().sum(),
    '% Nulo': (100 * df.isnull().sum()/df.shape[0]).round(1)
})

# Mostrar solo las columnas con algún valor nulo (opcional)
nulos_tbl_filtrado = nulos_tbl[nulos_tbl['Valores nulos'] > 0].sort_values('Valores nulos', ascending=False)

st.dataframe(nulos_tbl_filtrado, use_container_width=True)

st.markdown("""
<em>Nota: La tabla muestra el número y porcentaje de valores nulos para cada columna del DataFrame. Útil para orientar la limpieza de datos y las estrategias de imputación.</em>
""", unsafe_allow_html=True)

st.markdown("### 8. Total de valores faltantes")
total_faltantes = df.isnull().sum().sum()

if total_faltantes == 0:
    st.success("El DataFrame no tiene valores faltantes. ¡Está completamente limpio para el análisis!")
else:
    st.warning(f"El DataFrame tiene un total de {total_faltantes:,} valores faltantes (nulos) en todas las columnas.")
    # Opcional: muestra dónde están los nulos
    nulos_tbl = pd.DataFrame({
        'Columna': df.columns,
        'Valores nulos': df.isnull().sum()
    })
    st.dataframe(nulos_tbl[nulos_tbl['Valores nulos'] > 0].sort_values('Valores nulos', ascending=False), use_container_width=True)


st.markdown("### 9. Conteo de frecuencias de mecanismos causales")

conteo_mecanismos = df["mecanismo_causal"].value_counts().reset_index()
conteo_mecanismos.columns = ["Mecanismo causal", "Frecuencia"]

# Mostrar la tabla ordenada
st.dataframe(conteo_mecanismos, use_container_width=True)

# Análisis textual y visual del resultado principal
mayor_freq = conteo_mecanismos.iloc[0]
segundo_freq = conteo_mecanismos.iloc[1]

st.info(f'El mecanismo de suicidio más común es **"{mayor_freq[0]}"** con **{mayor_freq[1]:,} casos**, seguido por **"{segundo_freq[0]}"** con **{segundo_freq[1]:,} casos**. Otros mecanismos tienen frecuencias significativamente menores.')

# Opcional: gráfico de barras para visualización rápida
import altair as alt
chart = alt.Chart(conteo_mecanismos.head(10)).mark_bar().encode(
    x=alt.X('Frecuencia:Q'),
    y=alt.Y('Mecanismo causal:N', sort='-x')
).properties(
    width=600,
    height=350,
    title='Top 10 mecanismos causales de suicidio'
)
st.altair_chart(chart, use_container_width=True)

st.markdown("### 10. Distribución de suicidios por sexo")

# Buscar y procesar columna de sexo de manera robusta
col_sexo = next((c for c in df.columns if 'sexo' in c.lower()), None)
serie = (df[col_sexo].astype(str).str.strip().str.title()
         .replace({'Nan':'Desconocido'}))

vc = serie.value_counts(dropna=False)
pct = (vc / vc.sum() * 100).round(1)
tabla = pd.DataFrame({
    'Frecuencia Absoluta': vc,
    'Frecuencia Porcentual %': pct
}).reset_index().rename(columns={'index': 'Sexo'})

st.dataframe(tabla, use_container_width=True)

st.markdown("""
La distribución por sexo muestra que la mayoría de los casos de suicidio corresponden a hombres (**18,928 casos**), mientras que las mujeres representan una proporción menor (**4,616 casos**). Esta diferencia significativa sugiere que el género es un factor importante a considerar en el análisis de los suicidios en Colombia.
""")

st.markdown("### 11. Gráfica de la distribución por sexo")

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# El código de limpieza de la serie
col_sexo = next((c for c in df.columns if 'sexo' in c.lower()))
s = (df[col_sexo].astype(str).str.strip().str.title()
     .replace({'Nan':'Desconocido'}))
vc = s.value_counts(dropna=False)
pct = (vc / vc.sum() * 100).round(1)

vc = vc.sort_values(ascending=False)
pct = pct[vc.index]

fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(vc.index, vc.values, color=['#1f77b4', '#ff7f0e'])  # Colores académicos

ax.set_title("Distribución por sexo")
ax.set_ylabel("Número de casos")
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
plt.xticks(rotation=0)

# Etiquetas n y %
for rect, n, p in zip(bars, vc.values, pct.values):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
            f"{int(n):,}\n{p:.1f}%", ha='center', va='bottom', fontsize=10)

ax.margins(y=0.1)
plt.tight_layout()

st.pyplot(fig)

st.markdown("""
La gráfica muestra que la mayoría de los casos de suicidio corresponden a un solo sexo, lo que indica una distribución desigual entre hombres y mujeres. Este hallazgo resalta la importancia de considerar el género como factor relevante en el análisis epidemiológico de los suicidios en Colombia.
""")

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

st.markdown("### 12. Distribución de sexo por ciclo vital (porcentaje 100%)")

col_sexo  = next(c for c in df.columns if 'sexo' in c.lower())
col_ciclo = next(c for c in df.columns if 'ciclo' in c.lower())

sexo  = df[col_sexo].astype(str).str.strip().str.title().replace({'Nan':'Desconocido'})
ciclo = df[col_ciclo].astype(str).str.strip().str.title().replace({'Nan':'Desconocido'})

tab = pd.crosstab(ciclo, sexo)
rowpct = (tab.div(tab.sum(1), axis=0) * 100).round(1)
rowpct = rowpct.loc[tab.sum(1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10,6), dpi=120)
cats = rowpct.columns.tolist()
cmap = plt.cm.Blues_r
colors = [cmap(0.35 + 0.6*i/max(1, len(cats)-1)) for i in range(len(cats))]

rowpct.plot(kind='barh', stacked=True, ax=ax, color=colors, edgecolor='white', linewidth=0.8)

ax.set_xlim(0, 100)
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.grid(axis='x', linestyle='--', alpha=0.35)
ax.set_xlabel('Porcentaje (%)'); ax.set_ylabel('Ciclo vital')
ax.set_title('Distribución de sexo por ciclo vital (100%)')

for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(title='Sexo', bbox_to_anchor=(1.03,1), loc='upper left', frameon=False)

for i, (_, row) in enumerate(rowpct.iterrows()):
    cum = 0
    for val in row:
        if val >= 3:
            ax.text(cum + val/2, i, f'{val:.0f}%', ha='center', va='center', fontsize=9, color='#1f1f1f')
        cum += val

plt.tight_layout()
st.pyplot(fig)

st.markdown("""La distribución de suicidios en Colombia por sexo y ciclo vital entre 2015 y 2023 muestra una mayor prevalencia en hombres en todos los grupos de edad, con una brecha que se amplía progresivamente a medida que avanza la edad.""")

import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.markdown("### 13. Distribución de edad: histograma y densidad KDE")

# 1) Construcción robusta de la variable edad
s = df['edad_judicial'] 
edad = pd.to_numeric(s, errors='coerce')

if edad.isna().all():
    edad = s.astype(str).str.extract(r'(\d{1,3})')[0].astype(float)

# 2) Filtrar edades plausibles
edad = edad[(edad>=0) & (edad<=120)].dropna()
n = len(edad)

# 3) Visualización gráfica si hay datos suficientes
if n < 2:
    st.warning(f"Sin datos suficientes para densidad (n={n}). Revisa la variable 'edad_judicial'.")
else:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(edad, bins=30, density=True, alpha=0.5, color='#1f77b4', label='Histograma')
    edad.plot(kind='kde', lw=2, ax=ax, color='#ff7f0e', label='KDE')
    ax.set_xlabel('Edad (años)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de edad: histograma + densidad')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("""- La gráfica de densidad de Kernel (KDE) y el histograma normalizado muestran la distribución de edades de los casos de suicidio en Colombia entre 2015 y 2023. Muestra que la mayoría de los casos se concentran entre los 15 y 40 años, con un pico más fuerte alrededor de los 20 años. Después de los 40, la frecuencia disminuye gradualmente, siendo menos común en edades avanzadas. Es una distribución que refleja que el problema es más crítico en etapas tempranas y medias de la vida adulta.""")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.markdown("### 14. Detección de datos atípicos en edad mediante boxplot")

# Conversión y limpieza de la edad
edad = pd.to_numeric(df['edad_judicial'].astype(str).str.extract('(\d+)')[0], errors='coerce')
edad = edad[(edad >= 0) & (edad <= 120)].dropna()

if len(edad) < 2:
    st.warning("No hay suficientes datos válidos de edad para crear el boxplot.")
else:
    fig, ax = plt.subplots(figsize=(7,2.8))
    # Para aplicar color, usa patch_artist=True
    bp = ax.boxplot(edad, vert=False, showmeans=True, meanline=True, patch_artist=True)
    for box in bp['boxes']:
        box.set_facecolor('#d2e8f7')
    ax.set_xlabel('Edad (años)')
    ax.set_title('Boxplot de Edad (atípicos por IQR)')
    ax.set_xlim(0, edad.max() + 5)
    plt.tight_layout()
    st.pyplot(fig)

    # Estadísticas y conteo de atípicos por IQR
    q1, q3 = np.percentile(edad, [25, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_low = (edad < low).sum()
    n_high = (edad > high).sum()

    st.markdown(f"""
    **Estadísticas de edad:**
    - Q1: {q1:.1f}
    - Q3: {q3:.1f}
    - IQR: {iqr:.1f}
    - Límites para atípicos: [{low:.1f}, {high:.1f}]
    - Atípicos bajos: {n_low}
    - Atípicos altos: {n_high}
    - Total de registros válidos: {len(edad):,}
    """)

st.markdown("""
- Este boxplot muestra que la edad más frecuente de casos de suicidio se concentra entre los **20 y 50 años**, con una mediana (línea verde) alrededor de los **35 años**.
- También se pueden observar casos en edades más avanzadas, aunque son menos comunes.
- La caja representa el rango donde se encuentra la mayoría de los datos, mientras que las líneas extendidas indican la variabilidad total de las edades registradas.
""")

import numpy as np
import matplotlib.pyplot as plt
import textwrap as tw
import streamlit as st

st.markdown("### 15. Distribución de frecuencias de mecanismos causales")

COL = 'mecanismo_causal'  # si el nombre varía, ajústalo aquí

s = (df[COL].astype(str).str.strip()
       .where(lambda x: x.ne('nan'), 'Falta_dato'))
freq = s.value_counts(dropna=False).sort_values(ascending=True)
pct  = (freq/freq.sum()*100).round(1)

fig, ax = plt.subplots(figsize=(11,6), dpi=150)
bars = ax.barh(freq.index, freq.values, color='#6184A7')

# Etiquetas de frecuencia y porcentaje
for i, (v, p) in enumerate(zip(freq.values, pct.values)):
    ax.text(v, i, f'{v:,.0f}  ({p:.1f}%)', va='center', ha='left', fontsize=9)

ax.set_xlabel('Frecuencia')
ax.set_ylabel('Mecanismo causal')
ax.set_title('Mecanismo causal — Barras ordenadas por frecuencia')
ax.xaxis.set_major_formatter(lambda x,pos: f'{int(x):,}'.replace(',', '.'))

# Etiquetas largas — envuelve
ax.set_yticks(range(len(freq.index)))
ax.set_yticklabels(["\n".join(tw.wrap(l, 28)) for l in freq.index])

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
La gráfica nos permite identificar los mecanismos causales más comunes y menos comunes de suicidio en el conjunto de datos. Se observa que el método predominante es la **asfixia**, con **15,512 casos** (65.5% del total), seguido por el uso de **sustancias tóxicas** con **3,829 casos** (16.3%). Otros métodos como armas de fuego, objetos cortopunzantes y ahogamiento presentan frecuencias significativamente menores.
""")

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.markdown("### 16. Distribución de frecuencias de mecanismos causales — Lollipop")

COL = 'mecanismo_causal'

freq = (df[COL].astype(str).str.strip()
          .where(lambda x: x.ne('nan'), 'Falta_dato')
          .value_counts()
          .sort_values())

y = np.arange(len(freq))
fig, ax = plt.subplots(figsize=(11,6), dpi=150)

# Lollipop: líneas y puntos
ax.hlines(y, 0, freq.values, color='#6184A7')
ax.plot(freq.values, y, 'o', color='#E85D75', markersize=8)

ax.set_yticks(y)
ax.set_yticklabels(freq.index, fontsize=10)
ax.set_xlabel('Frecuencia')
ax.set_title('Mecanismo causal — Lollipop')
for i, v in enumerate(freq.values):
    ax.text(v, i, f' {v:,.0f}', va='center', ha='left', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap as tw
import streamlit as st

st.markdown("### 17. Distribución de mecanismos causales por sexo (Top-K)")

COL_MEC = 'mecanismo_causal'
COL_SEX = 'sexo_de_la_victima'
TOP_K   = 6

# Limpieza
df_ = df[[COL_SEX, COL_MEC]].dropna().copy()
df_[COL_SEX] = df_[COL_SEX].astype(str).str.strip().str.title()
df_[COL_MEC] = df_[COL_MEC].astype(str).str.strip().replace({'nan':'Falta_dato'})

# Top-K global + Otros
top = df_[COL_MEC].value_counts().index[:TOP_K].tolist()
df_['mec_plot'] = np.where(df_[COL_MEC].isin(top), df_[COL_MEC], 'Otros')

# % por sexo
tab = df_.value_counts([COL_SEX,'mec_plot']).unstack(fill_value=0)
rowpct = tab.div(tab.sum(1), axis=0)*100

# Selecciona los 2 sexos más frecuentes para panel limpio
sexos = tab.sum(1).sort_values(ascending=False).index[:2].tolist()
cats  = rowpct.sum(0).sort_values(ascending=False).index.tolist()

fig, axes = plt.subplots(1, len(sexos), figsize=(12,5), dpi=150, sharey=True)
if len(sexos) == 1: axes = [axes]

for ax, sx in zip(axes, sexos):
    vals = rowpct.loc[sx, cats]
    y    = np.arange(len(cats))
    ax.barh(y, vals.values, color='#4c78a8', edgecolor='black', linewidth=0.6)
    for i, v in enumerate(vals.values):
        if v >= 5:
            ax.text(v+0.5, i, f'{v:.0f}%', va='center', ha='left', fontsize=9)
    ax.set_title(f'{sx}  (n={int(tab.loc[sx].sum()):,})'.replace(',', '.'), fontsize=11)
    ax.set_xlim(0, 100); ax.set_xlabel('%')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=':', alpha=0.6)
axes[0].set_yticks(np.arange(len(cats)))
axes[0].set_yticklabels(["\n".join(tw.wrap(c, 20)) for c in cats], fontsize=10)
fig.suptitle('Mecanismo causal — Distribución 100% por sexo', fontsize=12)
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
La **asfixia** es el mecanismo predominante en ambos grupos (67% en hombres y 60% en mujeres). La principal divergencia se encuentra en los métodos secundarios: la **intoxicación por tóxicos** muestra una prevalencia marcadamente superior en mujeres (27%) frente a hombres (14%), mientras que el uso de **armas de fuego** constituye un mecanismo significativo casi exclusivamente en la población masculina (12%).

Estos patrones sugieren diferencias relevantes de género en los medios empleados y aportan información crítica para el diseño de intervenciones preventivas específicas.
""")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap as tw
import streamlit as st

st.markdown("### 18. Diferencias porcentuales de mecanismos causales por sexo")

COL_MEC = 'mecanismo_causal'
COL_SEX = 'sexo_de_la_victima'
TOP_K   = 10

df_ = df[[COL_SEX, COL_MEC]].dropna().copy()
df_[COL_SEX] = df_[COL_SEX].astype(str).str.strip().str.title()
df_[COL_MEC] = df_[COL_MEC].astype(str).str.strip().replace({'nan':'Falta_dato'})

# Toma dos sexos principales
sexos = df_[COL_SEX].value_counts().index[:2].tolist()
if len(sexos) < 2:
    st.error('Se requieren al menos dos categorías de sexo para la comparación.')
else:
    # % por sexo y mecanismo
    tab = df_.value_counts([COL_SEX, COL_MEC]).unstack(fill_value=0)
    rowpct = tab.div(tab.sum(1), axis=0)*100
    rowpct = rowpct.loc[sexos]

    # Top-K por suma promedio
    cats = rowpct.mean(0).sort_values(ascending=False).index[:TOP_K]
    D = (rowpct.loc[sexos[0], cats] - rowpct.loc[sexos[1], cats]).sort_values(key=lambda s: s.abs(), ascending=True)
    cats_ord = D.index.tolist()

    y = np.arange(len(cats_ord))
    fig, ax = plt.subplots(figsize=(10,6), dpi=150)

    a = rowpct.loc[sexos[0], cats_ord].values
    b = rowpct.loc[sexos[1], cats_ord].values

    # Líneas conectoras (dumbbell style)
    for i, (v1, v2) in enumerate(zip(a, b)):
        ax.plot([v1, v2], [i, i], '-', linewidth=2, color='#9da3a6')
    
    ax.plot(a, y, 'o', label=sexos[0], markersize=6, color='#4c78a8')
    ax.plot(b, y, 's', label=sexos[1], markersize=5, color='#f58518')

    for i, (v1, v2) in enumerate(zip(a, b)):
        ax.text(v1, i+0.2, f'{v1:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.text(v2, i-0.2, f'{v2:.0f}%', ha='center', va='top', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(["\n".join(tw.wrap(c, 28)) for c in cats_ord], fontsize=9)
    ax.set_xlabel('%'); ax.set_xlim(0, 100)
    ax.set_title('Mecanismo causal — Diferencia porcentual por sexo', fontsize=12)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("""
La distribución de los mecanismos causales revela una marcada **disparidad de género** en la elección del método. Si bien los **Generadores de asfixia** constituyen el principal mecanismo para ambos grupos, se observan diferencias proporcionales significativas: el mecanismo Tóxico se registra en mujeres con el **doble de frecuencia** que en hombres, mientras que la utilización de **Proyectil de arma de fuego** por parte de los hombres **supera en seis veces** la registrada en mujeres.

Estos contrastes sugieren la necesidad de intervenciones y prevención diferenciadas por género, considerando las realidades y riesgos específicos en cada grupo.
""")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap as tw
import streamlit as st

st.markdown("### 18. Diferencia porcentual de mecanismos causales por sexo")

COL_MEC = 'mecanismo_causal'
COL_SEX = 'sexo_de_la_victima'
TOP_K = 10

df_ = df[[COL_SEX, COL_MEC]].dropna().copy()
df_[COL_SEX] = df_[COL_SEX].astype(str).str.strip().str.title()
df_[COL_MEC] = df_[COL_MEC].astype(str).str.strip().replace({'nan': 'Falta_dato'})

# Toma dos sexos principales
sexos = df_[COL_SEX].value_counts().index[:2].tolist()
if len(sexos) < 2:
    st.error('Se requieren al menos dos categorías de sexo para la comparación.')
else:
    # % por sexo y mecanismo
    tab = df_.value_counts([COL_SEX, COL_MEC]).unstack(fill_value=0)
    rowpct = tab.div(tab.sum(1), axis=0) * 100
    rowpct = rowpct.loc[sexos]

    # Top-K por suma promedio
    cats = rowpct.mean(0).sort_values(ascending=False).index[:TOP_K]
    D = (rowpct.loc[sexos[0], cats] - rowpct.loc[sexos[1], cats]).sort_values(key=lambda s: s.abs(), ascending=True)
    cats_ord = D.index

    y = np.arange(len(cats_ord))
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    a = rowpct.loc[sexos[0], cats_ord].values
    b = rowpct.loc[sexos[1], cats_ord].values

    # Dumbbell lines y puntos
    for i, (v1, v2) in enumerate(zip(a, b)):
        ax.plot([v1, v2], [i, i], '-', linewidth=2, color='#9da3a6')
    ax.plot(a, y, 'o', label=sexos[0], markersize=6, color='#4c78a8')
    ax.plot(b, y, 's', label=sexos[1], markersize=5, color='#f58518')

    for i, (v1, v2) in enumerate(zip(a, b)):
        ax.text(v1, i+0.2, f'{v1:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.text(v2, i-0.2, f'{v2:.0f}%', ha='center', va='top', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(["\n".join(tw.wrap(c, 28)) for c in cats_ord], fontsize=9)
    ax.set_xlabel('%'); ax.set_xlim(0, 100)
    ax.set_title('Mecanismo causal — Diferencia porcentual por sexo', fontsize=12)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("""
La **distribución de los mecanismos causales** revela una marcada disparidad de género en la elección del método. Si bien los **Generadores de asfixia** constituyen el principal mecanismo para ambos grupos, se observan diferencias proporcionales significativas: el mecanismo **Tóxico** se registra en mujeres con el **doble de frecuencia** que en hombres, mientras que la utilización de **Proyectil de arma de fuego** por parte de los hombres **supera en seis veces** la registrada en mujeres.

Estos hallazgos refuerzan la importancia de diseñar estrategias de prevención diferenciadas y ponen en evidencia el impacto del género en la elección del método suicida.
""")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.markdown("### 19. Distribución de frecuencias por zona (Rural/Urbana)")

COL = 'zona_del_hecho'  # ajusta si tu columna difiere

s = (df[COL].astype(str).str.strip().str.title()
       .replace({'Nan': 'Falta_dato', 'N/a': 'Falta_dato'}))
# Normaliza etiquetas comunes
s = s.replace({'Urbano': 'Urbana', 'Rural': 'Rural'})

freq = s.value_counts(dropna=False).sort_values(ascending=True)
pct  = (freq/freq.sum()*100).round(1)

fig, ax = plt.subplots(figsize=(9,5), dpi=150)
ax.barh(freq.index, freq.values, edgecolor='black', linewidth=0.6, color='#6184A7')
for i, (v, p) in enumerate(zip(freq.values, pct.values)):
    ax.text(v, i, f'{v:,.0f}  ({p:.1f}%)'.replace(',', '.'),
            va='center', ha='left', fontsize=9)
ax.set_xlabel('Frecuencia'); ax.set_ylabel('Zona del hecho')
ax.set_title('Zona del hecho — Barras ordenadas (Urbana/Rural)')
ax.grid(axis='x', linestyle=':', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)

st.markdown("""La "Cabecera Municipal" (principal área urbana) concentra tres cuartas partes de todos los casos, con un 74.2% (17,466 hechos). En contraste, la "Parte Rural" y los "Centros Poblados" suman en conjunto el 25.2%, indicando una frecuencia considerablemente menor en áreas no urbanizadas.""")


import pandas as pd
import streamlit as st
import altair as alt

st.markdown("### 20. Evolución anual de casos del hecho")

# Conversión y limpieza
df['a_o_del_hecho'] = pd.to_numeric(df['a_o_del_hecho'], errors='coerce')
df_valid = df.dropna(subset=['a_o_del_hecho']).copy()
df_valid['a_o_del_hecho'] = df_valid['a_o_del_hecho'].astype(int)

# Conteo por año
casos_por_año = df_valid['a_o_del_hecho'].value_counts().sort_index()
df_linea = casos_por_año.reset_index()
df_linea.columns = ['Año', 'Casos']

chart = alt.Chart(df_linea).mark_line(point=alt.OverlayMarkDef(filled=True, fill='blue')).encode(
    x=alt.X('Año:O', axis=alt.Axis(labelAngle=0), sort='ascending'),
    y='Casos:Q',
    tooltip=['Año', 'Casos']
).properties(
    width=670,
    height=370,
    title='Tendencia Anual de Presuntos Suicidios en Colombia'
).interactive()

st.altair_chart(chart, use_container_width=True)

st.markdown("""
- La gráfica muestra la evolución anual de casos con un **claro crecimiento tendencial entre 2015 y 2018**, leve descenso en 2019, **caída marcada en 2020** y recuperación acelerada desde 2021 hasta alcanzar el **máximo en 2023**.
- Se observa un **mínimo de 2.068 casos en 2015** y un **máximo de 3.195 en 2023**, con **total n = 23.544**. La forma en V alrededor de 2020 sugiere un **choque exógeno o posible subregistro en ese año**; no se evalúa estacionalidad porque la unidad temporal es anual. 
- La tendencia posterior supera el nivel pre-choque, lo que indica **reanudación y aceleración** de la dinámica previa.
""")

import re, unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.markdown("### 21. Mapa de calor por departamento y tabla de participación (%)")

# ----- Detección de columna de departamento -----
dept_col_candidates = [
    'Departamento del hecho DANE','Depto_Nom','Departamento_ocurrencia',
    'Departamento','DEPARTAMENTO','Depto','departamento'
]
dept_col = next((c for c in dept_col_candidates if c in df.columns), None)
if dept_col is None:
    matches = [c for c in df.columns if 'depart' in c.lower()]
    if not matches:
        st.error("No se encontró columna de departamento.")
        st.stop()
    dept_col = matches[0]

# ----- Verifica columnas lat/lon -----
if not {'latitud','longitud'}.issubset(df.columns):
    st.error("Faltan columnas 'latitud' y 'longitud'")
    st.stop()

# Limpieza y agregación
df_clean = df[df[dept_col].astype(str).ne("Sin información")].copy()
map_data = (df_clean.groupby(dept_col, dropna=False)
            .agg(lat=('latitud', 'mean'),
                 lon=('longitud', 'mean'),
                 casos=(dept_col, 'size'))
            .reset_index())

# Normaliza nombres y tamaño de burbuja
def _title(s): 
    s = str(s).strip()
    s = re.sub(r'\b(de|del|la|y)\b', lambda m: m.group(0).lower(), s.title(), flags=re.I)
    return s
map_data['dept_clean'] = map_data[dept_col].map(_title)
map_data['size_norm']  = np.log1p(map_data['casos']) * 6

fig = px.scatter_mapbox(
    map_data,
    lat="lat", lon="lon",
    size="size_norm", size_max=28,
    color="casos",
    color_continuous_scale=["#E0F3F8", "#91BFDB", "#4575B4", "#313695"],
    hover_name="dept_clean",
    hover_data={"casos":":,", "lat":False, "lon":False, "size_norm":False},
    zoom=5, center=dict(lat=4.6, lon=-74.1),
    mapbox_style="open-street-map",
    title=f"Presuntos suicidios por departamento — Total: {map_data['casos'].sum():,} casos"
)
fig.update_layout(
    coloraxis_colorbar=dict(
        title="Número de casos",
        tickformat=",",
        len=0.75,
        thickness=15,
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    height=650, width=900,
    font=dict(size=12)
)
st.plotly_chart(fig, use_container_width=True)

# ----- Tablas PRO -----
tbl = (map_data
       .assign(Participación=lambda d: d["casos"] / d["casos"].sum())
       .sort_values("casos", ascending=False, ignore_index=True))
tbl.insert(0, "Rank", tbl.index + 1)
tbl = tbl.rename(columns={dept_col: "Departamento", "casos": "Casos"})
tbl['Participación (%)'] = (tbl['Participación']*100).round(1).astype(str)+' %'

st.markdown("**Tabla de participación porcentual por departamento:**")
st.dataframe(tbl[['Rank','Departamento','Casos','Participación (%)']], use_container_width=True)

top10 = tbl.head(10)
st.markdown("**Top 10 departamentos por casos:**")
st.dataframe(top10[['Rank','Departamento','Casos','Participación (%)']], use_container_width=True)

# Resumen
resumen = pd.DataFrame({
    "Total casos": [tbl["Casos"].sum()],
    "N departamentos": [len(tbl)],
    "Promedio casos/depto": [round(tbl["Casos"].mean(), 1)],
    "Mediana": [int(tbl["Casos"].median())],
    "Desviación estándar": [round(tbl["Casos"].std(), 1)]
})
st.markdown("**Resumen estadístico:**")
st.dataframe(resumen, use_container_width=True)

st.markdown("""## 2. Análisis Bivariado""")

import pandas as pd
import altair as alt
import streamlit as st
import numpy as np

st.markdown("#### 2.1 Tendencia anual de casos de suicidio (2015-2023)")

# Prepara los datos
df['a_o_del_hecho'] = pd.to_numeric(df['a_o_del_hecho'], errors='coerce')
df_valid = df.dropna(subset=['a_o_del_hecho']).copy()
df_valid['a_o_del_hecho'] = df_valid['a_o_del_hecho'].astype(int)

casos_por_año = df_valid['a_o_del_hecho'].value_counts().sort_index()
total = len(df_valid)
df_linea = casos_por_año.reset_index()
df_linea.columns = ['Año', 'Casos']
df_linea['Porcentaje'] = (df_linea['Casos']/total*100).round(1)

# Línea de tendencia lineal (regresión)
z = np.polyfit(df_linea['Año'], df_linea['Casos'], 1)
p = np.poly1d(z)
df_linea['Tendencia'] = p(df_linea['Año'])

# Grafico interactivo
base = alt.Chart(df_linea).encode(
    x=alt.X('Año:O', axis=alt.Axis(labelAngle=0), sort='ascending')
)
linea = base.mark_line(point=alt.OverlayMarkDef(filled=True, fill='royalblue'), color="royalblue", size=3).encode(
    y=alt.Y('Casos:Q', title='Número de Casos'),
    tooltip=['Año', 'Casos', alt.Tooltip('Porcentaje:Q', format='.1f')]
)
trend = base.mark_line(strokeDash=[5,3], color='red').encode(
    y='Tendencia:Q'
)
pts = base.mark_circle(size=95, color='royalblue').encode(
    y='Casos:Q'
)

labels = base.mark_text(align='center', dy=-20, fontSize=11).encode(
    y='Casos:Q',
    text=alt.Text('Casos:Q', format=',') 
)
perc_labels = base.mark_text(align='center', dy=0, fontSize=9, color='gray').encode(
    y=alt.Y('Casos:Q', stack=None),
    text=alt.Text('Porcentaje:Q', format='.1f')
)

# Anotación especial en 2020
highlight = alt.Chart(pd.DataFrame({
    'Año':[2020], 
    'Casos':df_linea[df_linea['Año']==2020]['Casos'].values,
    'Anotación':["Caída en 2020 (COVID-19)"]
})).mark_text(
    align='left', baseline='middle', dx=50, dy=-70, color='black', fontSize=12, fontWeight='bold'
).encode(
    x='Año:O',
    y='Casos:Q',
    text='Anotación:N'
)

chart = (linea + pts + trend + labels + perc_labels + highlight).properties(
    width=750, height=370,
    title='Evolución Anual de Casos de Suicidio'
).interactive()

st.altair_chart(chart, use_container_width=True)

st.markdown("""- El análisis de la tendencia anual de los casos de suicidio en Colombia entre 2015 y 2023 revela un patrón dinámico con fluctuaciones significativas a lo largo del periodo.""")

import altair as alt
import pandas as pd
import streamlit as st

st.markdown("#### 2.2 Top 10 Departamentos con Más Casos")

# Asegúrate de que la columna tiene el nombre correcto
col_dpto = 'departamento_del_hecho_dane'
top_10_dpto = df[col_dpto].value_counts().head(10)

df_top10 = pd.DataFrame({
    'Departamento': top_10_dpto.index,
    'Casos': top_10_dpto.values
})
df_top10['Porcentaje'] = (df_top10['Casos']/df_top10['Casos'].sum()*100).round(1)

bars = alt.Chart(df_top10).mark_bar(
    color='royalblue'
).encode(
    x=alt.X('Casos:Q', title='Cantidad de Casos'),
    y=alt.Y('Departamento:N', sort='-x', title='Departamento')
)

text = alt.Chart(df_top10).mark_text(
    align='left',
    baseline='middle',
    dx=3,
    fontSize=13
).encode(
    x=alt.X('Casos:Q'),
    y=alt.Y('Departamento:N', sort='-x'),
    text=alt.Text('label:N')
)

df_top10['label'] = df_top10['Casos'].astype(str) + " (" + df_top10['Porcentaje'].astype(str) + "%)"

chart = (bars + text).properties(
    width=600,
    height=365,
    title='Top 10 Departamentos con Más Casos'
)

st.altair_chart(chart, use_container_width=True)

st.markdown("""- Las tres primeras regiones, Antioquia (24,5%) , Bogotá (21,1%) y Valle del Cauca (12,9%) , acumulan en conjunto casi el 60% del total. A partir de ahí, las cifras descienden notablemente en los demás departamentos del top 10. Se observa que Atlántico se encuentra en la octava posición de esta clasificación, con 809 casos, representando el 5.1% del total mostrado.""")

import re
import pandas as pd
import altair as alt
import streamlit as st

st.markdown("#### 2.3 Casos por Grupo de Edad de la Víctima")

# --- Preparar y ordenar ---
def edad_key(x):
    m = re.search(r'\d+', str(x))
    return int(m.group()) if m else 999

# Reemplaza con tu columna exacta
col_edad = 'grupo_de_edad_de_la_victima'
order = sorted(df[col_edad].unique(), key=edad_key)
df_edad = df[col_edad].value_counts().reindex(order).reset_index()
df_edad.columns = ['Grupo de Edad', 'Casos']

total = df_edad['Casos'].sum()
df_edad['Porcentaje'] = (df_edad['Casos']/total*100).round(1)
df_edad['label'] = df_edad['Casos'].astype(str) + " (" + df_edad['Porcentaje'].astype(str) + "%)"

bars = alt.Chart(df_edad).mark_bar(color='royalblue').encode(
    x=alt.X('Casos:Q', title='Cantidad de Casos'),
    y=alt.Y('Grupo de Edad:N', sort=order, title='Grupo de Edad')
)

text = alt.Chart(df_edad).mark_text(
    align='left', baseline='middle', dx=3, fontSize=12
).encode(
    x='Casos:Q',
    y=alt.Y('Grupo de Edad:N', sort=order),
    text=alt.Text('label:N')
)

chart = (bars + text).properties(
    width=740, height=365,
    title='Casos por Grupo de Edad de la Víctima'
)

st.altair_chart(chart, use_container_width=True)

st.markdown("""
La gráfica concluye que la **mayor cantidad de casos se concentra en la población adulta joven**, con el grupo de **20 a 24 años** siendo el más afectado en términos absolutos, seguido por los grupos de **25 a 29 años** y **30 a 34 años**. A partir de los **35 años**, la cantidad de casos disminuye progresivamente con la edad, aunque la afectación sigue siendo notoria en todos los grupos adultos.
""")

import re
import pandas as pd
import altair as alt
import streamlit as st

st.markdown("#### 2.4 Casos por Grupo de Edad y Sexo")

def edad_key(x):
    m = re.search(r'\d+', str(x))
    return int(m.group()) if m else 999

col_edad = 'grupo_de_edad_de_la_victima'
col_sexo = 'sexo_de_la_victima'

order = sorted(df[col_edad].unique(), key=edad_key)
df_sexo_edad = (
    df.groupby([col_edad, col_sexo])
    .size()
    .reset_index(name='Casos')
    .sort_values(by=col_edad, key=lambda s: s.map(edad_key))
)
df_sexo_edad['Porcentaje'] = (df_sexo_edad['Casos'] / df_sexo_edad['Casos'].sum() * 100).round(1)
df_sexo_edad['label'] = df_sexo_edad['Casos'].astype(str) + " (" + df_sexo_edad['Porcentaje'].astype(str) + "%)"

bars = alt.Chart(df_sexo_edad).mark_bar().encode(
    x=alt.X('Casos:Q', title='Cantidad de Casos'),
    y=alt.Y(f'{col_edad}:N', sort=order, title='Grupo de Edad'),
    color=alt.Color(f'{col_sexo}:N', scale=alt.Scale(domain=['Masculino','Femenino'], range=['#1f77b4','#9edae5']), title='Sexo')
)

text = alt.Chart(df_sexo_edad).mark_text(
    align='left', baseline='middle', dx=3, fontSize=10
).encode(
    x='Casos:Q',
    y=alt.Y(f'{col_edad}:N', sort=order),
    detail=alt.Detail(f'{col_sexo}:N'),
    text=alt.Text('label:N')
)

chart = (bars + text).properties(
    width=780, height=420,
    title='Casos por Grupo de Edad y Sexo'
)

st.altair_chart(chart, use_container_width=True)

st.markdown("""
La gráfica concluye que la **mayor cantidad de casos se concentra en la población masculina adulta joven**, siendo el grupo de **20 a 24 años** el más afectado en términos absolutos. La distribución revela una **mayor prevalencia de casos en hombres en todas las cohortes de edad**, lo que subraya una **disparidad de género consistente** en la muestra analizada.
""")

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import streamlit as st

def norm(s):
    s = str(s).strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return s.replace(' ', '_')

st.markdown("#### 2.5 Edad por sexo — Boxplot comparativo")

# Normalizar nombres de columnas
df.columns = [norm(c) for c in df.columns]

sexo_col = next(col for col in df.columns if 'sexo' in col)
edad_col = next(col for col in df.columns if 'edad' in col and 'grupo' not in col)

# Limpiar y preparar base
base = df[[sexo_col, edad_col]].copy()
base.columns = ['sexo', 'edad']
base['sexo'] = base['sexo'].astype(str).str.strip().str.title()
base['edad'] = pd.to_numeric(base['edad'].astype(str).str.extract('(\d+)')[0], errors='coerce')
base = base[(base['edad']>=0) & (base['edad']<=100)].dropna()

# Crear grupos de edad por sexo
grupos_dict = {s: base[base['sexo']==s]['edad'].values for s in base['sexo'].unique() if len(base[base['sexo']==s]) > 0}
labels = [f"{k}  (n={len(v):,})" for k,v in grupos_dict.items()]
grupos = list(grupos_dict.values())
meds = [np.median(v) for v in grupos]

fig, ax = plt.subplots(figsize=(7,3.8))
ax.boxplot(grupos, 
          vert=False, 
          labels=labels,
          showmeans=True,
          meanline=True,
          notch=True,
          showfliers=False,
          patch_artist=True,
          boxprops=dict(facecolor="#e9eef3", edgecolor="#2b2b2b"),
          medianprops=dict(color="#1a1a1a", linewidth=2),
          meanprops=dict(color="#666666", linewidth=1.2))

ax.set_xlabel("Edad (años)")
ax.set_ylabel("Sexo")
ax.set_xlim(0, 90)
ax.grid(axis="x", alpha=0.25)
ax.set_title("Edad por sexo — Presuntos suicidios, Colombia 2015–2023", pad=8)

# Etiquetas para medianas
for y, m in enumerate(meds, start=1):
    ax.text(m, y+0.12, f"med={m:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
st.pyplot(fig)

# Tabla de estadísticos resumen
resumen = []
for sex, ages in grupos_dict.items():
    resumen.append({
        'Sexo': sex,
        'n': len(ages),
        'Media': np.mean(ages),
        'Mediana': np.median(ages),
        'Desv. Est.': np.std(ages),
        'Mínimo': np.min(ages),
        'Máximo': np.max(ages)
    })
res_df = pd.DataFrame(resumen)
res_df = res_df[['Sexo','n','Mediana','Media','Desv. Est.','Mínimo','Máximo']].round(1)
st.markdown("**Estadísticos resumidos por sexo:**")
st.dataframe(res_df, use_container_width=True)

st.markdown("""
El **Boxplot segmentado por sexo** permite comparar la distribución de edades entre hombres y mujeres en los casos de suicidio en Colombia entre 2015 y 2023.

El análisis revela una **clara diferencia en la edad de afectación según el sexo**. La **edad mediana de los casos masculinos es notablemente superior** (35 años), y su 50% central se extiende hasta los 50 años. Por el contrario, la edad mediana en mujeres es **diez años más joven** (25 años), y su 50% central se concentra en un rango más compacto (aproximadamente 19 a 40 años).
""")

import pandas as pd
import streamlit as st

st.markdown("#### 2.6 Razones más comunes del suicidio")

# Prepara la tabla top-10
col_razon = 'razon_del_suicidio'
top_10 = df[col_razon].value_counts().head(10)
total = df[col_razon].notna().sum()

df_top10 = pd.DataFrame({
    'Razón del suicidio': top_10.index,
    'Casos': top_10.values
})
df_top10['Porcentaje'] = (df_top10['Casos'] / total * 100).round(1).astype(str) + " %"

st.dataframe(df_top10, use_container_width=True)

