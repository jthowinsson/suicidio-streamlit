import streamlit as st

# Imagen en la barra lateral
st.sidebar.image("images/Logo.png", use_container_width=True)

st.title("Suicidio en Colombia (2015 - 2023)")
st.subheader("Modelo de Clasificación XGBoost para el Mecanismo Causal de Suicidio en Colombia (2015–2023): Un Enfoque Predictivo")

st.write("Bienvenido, usa el menú lateral para navegar...")

st.sidebar.info("Siempre puedes hablar • Proyecto Dataviz ©2025")

st.header("Aproximación Conceptual")

# Imagen conceptual
st.image("images/Salud.png", caption="Imagen creada por IA Canva", use_container_width=True)

st.markdown("""
La conducta suicida es un fenómeno complejo, con diferentes implicaciones, tanto psicopatológicas como existenciales, sociales y morales, por lo que resulta complicado dar una definición única y universal a la misma. Diversos autores han tratado de definir la conducta suicida (Durkheim, Schneider), pero es tal vez Rojas (1984) el que establece un concepto más operativo:

> “Se entiende por suicidio aquella conducta o conjunto de conductas que, dirigidas por el propio sujeto, conducen a la muerte (suicidio consumado) o a una situación de gravedad mortal (suicidio frustrado), bien de forma activa o pasiva”.

Es decir, que dentro de la conducta suicida no sólo hay que contemplar la consumación del suicidio, sino también la cantidad de matices autoagresivos existentes en la misma y que necesariamente no llevan a la muerte a la persona pero que marcan a partir de este momento su propia existencia. 

Dos son los elementos que integran a la conducta suicida, el “Criterio auto infligido”, es decir la propia acción violenta, y el “Criterio de propósito”, que hace referencia a la finalidad de muerte. La presencia de ambos criterios o la ausencia de uno de ellos determinan las diferentes formas con que puede presentarse la conducta suicida.
""")

# Imagen sobre la conducta auto infligida
st.image("images/Conducta_AutoInflingido.jpeg", caption="Representación de Conducta AutoInfligida", use_container_width=True)

st.markdown("""
**Parasuicidio:**  
También llamado “Gesto Suicida”. Es el conjunto de conductas voluntarias e intencionales que el sujeto pone en marcha con el fin de producirse daño físico y cuyas consecuencias son el dolor, la desfiguración, la mutilación o el daño de alguna función o parte de su cuerpo, pero sin la intención de acabar con su vida.
Se incluyen aquí entre otros, los cortes en las muñecas, las sobredosis de medicamentos sin intención de muerte y las quemaduras. La intención en el Parasuicidio o Gesto Suicida no es por lo tanto la muerte, sino que tiene que ver con el deseo de conseguir algo (más cariño, que la pareja no le abandone, un empleo, etc.) para lo cual la persona cree que no dispone de otro tipo de recursos personales. 

**Ideas suicidas:**  
La persona contempla el suicidio como solución real a sus problemas, si bien aún no se ha producido un daño físico contra sí mismo. No existe aún una idea clara ni de cómo ni de cuando, pero entre las alternativas que puede tener para solucionar su situación problemática ya está presente el suicidio. 

**Crisis suicida:**  
De entre todas las alternativas que la persona disponía para solucionar la situación problemática, el suicidio comienza a tomar protagonismo. La idea ha tomado cuerpo y se activan a nivel psíquico un conjunto de impulsos de muerte, que le llevan a establecer un plan suicida.

**Suicidio consumado:**  
Cuando el Criterio de Propósito o de Muerte y el Criterio Auto Infligido se suman, se establece un plan de acción con diferentes niveles de elaboración. Si la puesta en práctica de este plan tiene “éxito” conduce a la muerte del sujeto.

**Suicidio frustrado:**  
Es un acto suicida que no conlleva a la muerte de la persona porque determinadas circunstancias externas, muchas veces casuales y siempre no previstas acontecen en el momento crítico. No es por lo tanto un Parasuicidio, ya que en el Suicidio Frustrado sí que hay una voluntad real de producirse la propia muerte. Sirva como ejemplo el caer sobre las cuerdas de un tendedero al arrojarse por una ventana.

**Tentativa de suicidio:**  
Toda conducta que busca la propia muerte pero para lograr el propósito la persona no emplea los medios adecuados y por lo tanto el sujeto no consigue acabar con su vida. Es un intento que puede fallar por múltiples causas, desde no tener una firme decisión de suicidarse hasta por el empleo de medios “blandos”. Existe el “propósito de muerte” pero el “criterio auto inflingido” no es el adecuado.
""")
