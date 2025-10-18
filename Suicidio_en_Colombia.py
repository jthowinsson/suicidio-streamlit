import streamlit as st

# Imagen en la barra lateral
st.sidebar.image("images/Logo.png", use_container_width=True)

# Título de la página

st.header("Introducción")

# Imagen de portada con leyenda
st.image("images/Portada.png", caption="Campaña de prevención del suicidio, Colombia.", use_container_width=True)

st.markdown("""
El suicidio es el acto deliberado mediante el cual una persona decide poner fin a su propia vida, constituyendo una problemática compleja que integra dimensiones psicológicas, sociales, culturales y biológicas. No existe una definición única y universal, pero en el ámbito de la salud pública y la psicología se reconoce como una acción autoinfligida con el propósito directo de provocar la muerte, bien sea de manera activa o pasiva [1,4,2].

El suicidio es uno de los principales problemas de salud pública a nivel mundial y está íntimamente vinculado con la salud mental. La evidencia científica señala que la mayoría de los suicidios están precedidos por trastornos mentales, especialmente depresión, trastorno bipolar, esquizofrenia, trastornos de personalidad y abuso de sustancias psicoactivas. De hecho, hasta el 95% de quienes mueren por suicidio presentan un diagnóstico psiquiátrico previo [3,4].

Según la Organización Mundial de la Salud, más de 720,000 personas fallecen anualmente por esta causa, ubicándose como una de las principales causas de muerte en adolescentes y adultos jóvenes [1,2]. Los intentos de suicidio y la ideación suicida — pensamientos recurrentes de autolesión o muerte — forman parte del espectro del comportamiento suicida y tienen una base principalmente psicológica y emocional, relacionada con alteraciones en la salud mental.

Existen múltiples factores de riesgo que contribuyen al suicidio: aislamiento social, desesperanza, problemas económicos, crisis familiares o personales y falta de apoyo emocional, todos estrechamente asociados con la desestabilización de la salud mental. La literatura reporta que el suicidio está rodeado de estigma y tabúes sociales, dificultando su prevención. Sin embargo, debe entenderse como un evento prevenible, abordando adecuadamente la salud mental mediante políticas integrales, intervenciones basadas en evidencia y fortalecimiento de factores protectores en la comunidad [2,1,3].

Por su magnitud y multidimensionalidad, el suicidio exige un enfoque científico, interdisciplinario y ético desde la perspectiva de la salud mental y la salud pública.
""")

# Subtítulo y contexto específico de Colombia

st.subheader("Contexto del suicidio en Colombia")

st.markdown("""
En Colombia, el suicidio ha surgido como una preocupación creciente en la última década, reflejando una realidad social y de salud mental que afecta especialmente a jóvenes y poblaciones vulnerables. Entre los años 2015 y 2023, los datos oficiales registraron un aumento en las tasas de suicidio, con un enfoque particular en grupos como los jóvenes de 18 a 28 años y comunidades indígenas, donde las dificultades para acceder a servicios de salud mental agravan el problema [5].

Las cifras oficiales de Colombia resaltan que este fenómeno no solo es un reflejo de problemas individuales, sino que también está influenciado por factores sociales como la violencia, la desigualdad, el desplazamiento forzado y las secuelas económicas y emocionales de la pandemia. Además, la estigmatización alrededor de la salud mental y el suicidio dificulta la búsqueda de ayuda y la implementación efectiva de políticas de prevención.

Este contexto colombiano particular, caracterizado por complejidades sociales y desigualdades regionales, refuerza la necesidad de estudios que permitan explicar y predecir el suicidio dentro del país, para así apoyar estrategias públicas y comunitarias más sensibles y eficaces.

Esta visión desde los datos oficiales y el contexto social permite anclar tu análisis en la realidad colombiana actual, subrayando la importancia de modelos estadísticos que puedan contribuir a la prevención y atención personalizada en salud mental.
""")

st.subheader("Objetivo general")

st.markdown("""
Estimar la asociación entre características sociodemográficas, contextuales y la probabilidad de que el mecanismo de suicidio sea “generadores de asfixia” (vs otros mecanismos) en Colombia durante 2015–2023.
""")

st.markdown("### Objetivos específicos")

st.markdown("""
- Caracterizar los mecanismos de suicidio registrados en Colombia entre 2015 y 2023 según variables sociodemográficas y contextuales.
- Construir una variable dependiente binaria para analizar el mecanismo predominante (generadores de asfixia vs otros).
- Estimar un modelo de regresión logística binaria para identificar factores asociados al uso de generadores de asfixia como mecanismo de suicidio.
- Evaluar el ajuste y desempeño del modelo mediante métricas de discriminación y calibración.
- Explorar la robustez del modelo a través de análisis de sensibilidad e interacciones entre variables clave.
""")
