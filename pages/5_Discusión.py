import streamlit as st

# Imagen en la barra lateral
st.sidebar.image("images/Logo.png", use_container_width=True)

st.markdown("""
## Discusión

- Los resultados subrayan la **viabilidad del enfoque predictivo contextual**.
- El AUC de **0.72** invita a incluir *variables adicionales* (historial clínico, redes de apoyo, intentos previos) para mejorar la capacidad explicativa.

- La **elevada importancia del municipio y grupo de edad** sugiere la necesidad de:
  - Políticas de prevención focalizadas
  - Campañas específicas por regiones

- La **limitación principal** reside en el *desbalanceo estructural de las clases*, que puede mitigarse con:
  - Técnicas de sobremuestreo (**SMOTE**)
  - Algoritmos de ensemble más complejos
""")

st.markdown("""
## Recomendaciones

1. **Mejoras en la Recolección de Datos**
   - Incluir variables sobre el historial clínico de salud mental
   - Documentar intentos previos de suicidio
   - Registrar información sobre redes de apoyo social
   - Incorporar datos sobre factores socioeconómicos

2. **Intervenciones Focalizadas**
   - Ajustar hiperparámetros mediante búsqueda sistemática
   - Desarrollar programas de prevención específicos por región
   - Adaptar estrategias según grupos de edad identificados como vulnerables
   - Fortalecer la capacitación del personal de salud en áreas de alto riesgo
   - Implementar sistemas de alerta temprana en municipios prioritarios

3. **Monitoreo y Evaluación**
   - Establecer métricas de seguimiento para evaluar intervenciones
   - Realizar actualizaciones periódicas del modelo con nuevos datos
   - Desarrollar dashboards para visualización de predicciones en tiempo real
   - Implementar sistemas de retroalimentación con expertos en el campo

4. **Consideraciones Éticas y de Privacidad**
   - Establecer protocolos estrictos de manejo de datos sensibles
   - Garantizar el anonimato en la recolección y procesamiento de información
   - Desarrollar guías éticas para el uso de predicciones en intervenciones
   - Mantener un enfoque centrado en la prevención y el apoyo
""")

st.markdown("""
## Conclusiones

1. El modelo **XGBoost desarrollado demuestra una capacidad predictiva moderada pero significativa** (AUC = 0.721) para identificar el mecanismo de suicidio, específicamente en casos de asfixia, superando el umbral de predicción aleatoria.

2. Las **variables geográficas y demográficas**, particularmente el municipio y el grupo de edad, emergen como los **predictores más relevantes**, sugiriendo patrones espaciales y etarios significativos en la elección del mecanismo de suicidio.

3. El análisis revela la **importancia de un enfoque territorialmente diferenciado** en las estrategias de prevención, considerando las particularidades de cada región y grupo poblacional.

4. Las **limitaciones actuales del modelo**, principalmente relacionadas con el desbalanceo de clases y la ausencia de variables psicosociales importantes, señalan **áreas claras de mejora para futuras iteraciones**.

5. La implementación de este modelo, junto con las recomendaciones propuestas, podría contribuir significativamente a:
   - Mejorar la identificación temprana de casos de alto riesgo
   - Optimizar la asignación de recursos preventivos
   - Desarrollar intervenciones más efectivas y focalizadas
   - Fortalecer los sistemas de vigilancia epidemiológica

6. El estudio **demuestra el potencial del aprendizaje automático** como herramienta complementaria en la comprensión y prevención del suicidio, siempre que se integre con **conocimiento experto y consideraciones éticas apropiadas**.
""")
