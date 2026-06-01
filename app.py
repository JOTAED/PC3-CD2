import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
import spacy

from transformers import pipeline

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="PLN Universidades Peruanas",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# CARGA DE MODELOS
# =====================================================

@st.cache_resource
def cargar_modelos():

    modelo_nb = joblib.load("modelo_sentimiento.pkl")

    vectorizador_tfidf = joblib.load(
        "vectorizador_tfidf.pkl"
    )

    modelo_lda = joblib.load(
        "modelo_lda.pkl"
    )

    vectorizador_lda = joblib.load(
        "vectorizador_lda.pkl"
    )

    return (
        modelo_nb,
        vectorizador_tfidf,
        modelo_lda,
        vectorizador_lda
    )


@st.cache_resource
def cargar_spacy():

    return spacy.load(
        "es_core_news_sm"
    )


@st.cache_resource
def cargar_transformer():

    return pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis"
    )


(
    modelo_nb,
    vectorizador_tfidf,
    modelo_lda,
    vectorizador_lda
) = cargar_modelos()

nlp = cargar_spacy()

modelo_transformer = cargar_transformer()

# =====================================================
# DATASET
# =====================================================

@st.cache_data
def cargar_dataset():

    return pd.read_csv(
        "universidades_limpio.csv"
    )


df = cargar_dataset()

# =====================================================
# TEMAS LDA
# =====================================================

temas = {
    0: "📚 Calidad académica e infraestructura",
    1: "🌳 Ambiente universitario",
    2: "🔬 Investigación y prestigio",
    3: "🎓 Experiencia estudiantil",
    4: "🏢 Servicios y atención"
}

# =====================================================
# TITULO
# =====================================================

st.title(
    "🎓 Analizador Inteligente de Opiniones Universitarias"
)

st.markdown("""
Proyecto desarrollado con técnicas de Procesamiento de Lenguaje Natural (PLN).

### Técnicas utilizadas

✅ TF-IDF

✅ Clasificación de Sentimientos

✅ Modelo Naive Bayes

✅ Modelo Preentrenado en Español

✅ Reconocimiento de Entidades (NER)

✅ Topic Modeling (LDA)

✅ Visualización Interactiva
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Filtros")

universidad = st.sidebar.selectbox(
    "Seleccione universidad",
    ["Todas"] + list(df["name_place"].unique())
)

if universidad == "Todas":
    df_filtrado = df.copy()
else:
    df_filtrado = df[
        df["name_place"] == universidad
    ]

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "😊 Análisis Completo",
    "🏷️ Entidades",
    "📚 Temas",
    "📊 Dashboard",
    "📂 Cargar Archivo"
])

# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.header(
        "Análisis Inteligente de Opiniones"
    )

    texto = st.text_area(
        "Ingrese una opinión:"
    )

    if st.button(
        "Analizar opinión"
    ):

        if texto.strip() != "":

            # =========================
            # NAIVE BAYES
            # =========================

            texto_vect = vectorizador_tfidf.transform(
                [texto]
            )

            pred_nb = modelo_nb.predict(
                texto_vect
            )[0]

            # =========================
            # TRANSFORMER
            # =========================

            resultado = modelo_transformer(
                texto
            )[0]

            etiqueta = resultado["label"]

            score = resultado["score"]

            if etiqueta.upper() in [
                "POS",
                "POSITIVE"
            ]:
                sentimiento_transformer = "Positivo"

            elif etiqueta.upper() in [
                "NEG",
                "NEGATIVE"
            ]:
                sentimiento_transformer = "Negativo"

            else:
                sentimiento_transformer = "Neutral"

            # =========================
            # TOPIC MODELING
            # =========================

            texto_topic = vectorizador_lda.transform(
                [texto]
            )

            probabilidades = modelo_lda.transform(
                texto_topic
            )[0]

            tema_predicho = np.argmax(
                probabilidades
            )

            # =========================
            # RESULTADOS
            # =========================

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Modelo Entrenado"
                )

                if pred_nb == "Positivo":
                    st.success(pred_nb)

                elif pred_nb == "Negativo":
                    st.error(pred_nb)

                else:
                    st.warning(pred_nb)

            with col2:

                st.subheader(
                    "Modelo Preentrenado"
                )

                if sentimiento_transformer == "Positivo":
                    st.success(
                        sentimiento_transformer
                    )

                elif sentimiento_transformer == "Negativo":
                    st.error(
                        sentimiento_transformer
                    )

                else:
                    st.warning(
                        sentimiento_transformer
                    )

                st.metric(
                    "Confianza",
                    f"{score*100:.2f}%"
                )

            st.divider()

            st.subheader(
                "Tema Detectado"
            )

            st.info(
                temas[tema_predicho]
            )

            # =========================
            # GRAFICO DE TEMAS
            # =========================

            df_topics = pd.DataFrame({

                "Tema": list(
                    temas.values()
                ),

                "Probabilidad":
                probabilidades

            })

            st.subheader(
                "Probabilidad por Tema"
            )

            st.bar_chart(
                df_topics.set_index(
                    "Tema"
                )
            )

            # =========================
            # NER AUTOMÁTICO
            # =========================

            st.subheader(
                "Entidades Detectadas"
            )

            doc = nlp(texto)

            entidades = []

            for ent in doc.ents:

                entidades.append(
                    [
                        ent.text,
                        ent.label_
                    ]
                )

            if len(entidades):

                df_ent = pd.DataFrame(
                    entidades,
                    columns=[
                        "Entidad",
                        "Tipo"
                    ]
                )

                st.dataframe(
                    df_ent
                )

            else:

                st.info(
                    "No se encontraron entidades."
                )

# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.header(
        "Reconocimiento de Entidades"
    )

    texto_ner = st.text_area(
        "Ingrese texto:",
        key="ner"
    )

    if st.button(
        "Detectar Entidades"
    ):

        doc = nlp(
            texto_ner
        )

        entidades = []

        for ent in doc.ents:

            entidades.append(
                [
                    ent.text,
                    ent.label_
                ]
            )

        if len(entidades):

            st.dataframe(
                pd.DataFrame(
                    entidades,
                    columns=[
                        "Entidad",
                        "Tipo"
                    ]
                )
            )

        else:

            st.warning(
                "No se detectaron entidades."
            )

# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.header(
        "Temas Descubiertos por LDA"
    )

    for tema in temas.values():
        st.write(tema)

# =====================================================
# TAB 4 DASHBOARD
# =====================================================

with tab4:

    st.header(
        "Dashboard General"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Opiniones",
        len(df_filtrado)
    )

    c2.metric(
        "Universidades",
        df_filtrado[
            "name_place"
        ].nunique()
    )

    c3.metric(
        "Rating Promedio",
        round(
            df_filtrado[
                "rating"
            ].mean(),
            2
        )
    )

    st.divider()

    # ------------------------

    st.subheader(
        "Distribución de Sentimientos"
    )

    fig1, ax1 = plt.subplots()

    sns.countplot(
        data=df_filtrado,
        x="sentimiento",
        ax=ax1
    )

    st.pyplot(fig1)

    # ------------------------

    st.subheader(
        "Distribución de Ratings"
    )

    fig2, ax2 = plt.subplots()

    sns.histplot(
        df_filtrado["rating"],
        bins=5,
        ax=ax2
    )

    st.pyplot(fig2)

    # ------------------------

    st.subheader(
        "Opiniones por Universidad"
    )

    fig3, ax3 = plt.subplots(
        figsize=(8,4)
    )

    df_filtrado[
        "name_place"
    ].value_counts().plot(
        kind="bar",
        ax=ax3
    )

    st.pyplot(fig3)

    # ------------------------

    st.subheader(
        "WordCloud"
    )

    texto_total = " ".join(
        df_filtrado[
            "texto_limpio"
        ].dropna().astype(str)
    )

    nube = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(
        texto_total
    )

    fig4, ax4 = plt.subplots(
        figsize=(12,6)
    )

    ax4.imshow(nube)

    ax4.axis("off")

    st.pyplot(fig4)

# =====================================================
# TAB 5
# =====================================================

with tab5:

    st.header(
        "Carga de Archivos"
    )

    archivo = st.file_uploader(
        "Seleccione CSV",
        type=["csv"]
    )

    if archivo:

        datos = pd.read_csv(
            archivo
        )

        st.success(
            "Archivo cargado correctamente"
        )

        st.write(
            "Dimensiones:",
            datos.shape
        )

        st.dataframe(
            datos.head()
        )