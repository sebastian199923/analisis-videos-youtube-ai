import streamlit as st
import numpy as np
import google.generativeai as genai
import yt_dlp
import subprocess
import re
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import tempfile

st.set_page_config(page_title="Análisis Inteligente YouTube", page_icon="▶️")


def obtener_subtitulos(url):
    comando = [
        "yt-dlp",
        "--write-auto-sub",
        "--sub-lang", "es",
        "--skip-download",
        "--convert-subs", "srt",
        url
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(comando, check=True, cwd=temp_dir,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        archivo_srt = next((archivo for archivo in os.listdir(
            temp_dir) if archivo.endswith(".es.srt")), None)
        if not archivo_srt:
            return "No se encontraron subtítulos en español."

        with open(f"{temp_dir}/{archivo_srt}", "r", encoding="utf-8") as f:
            contenido = f.read()
    return limpiar_subtitulos(contenido)


# Función para limpiar subtítulos de la trascripcion obtenida
def limpiar_subtitulos(texto):
    lineas = texto.split("\n")
    texto_limpio = []
    ultima_linea = ""

    for linea in lineas:
        linea = re.sub(r"^\d+$", "", linea)  # Eliminar números de índice
        # Eliminar timestamps
        linea = re.sub(
            r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", linea)
        linea = linea.strip()

        if linea and linea != ultima_linea:
            texto_limpio.append(linea)
            ultima_linea = linea

    return " ".join(texto_limpio)


# Función para fragmentar texto
def fragmentar_texto(transcripcion: str):
    splitter = CharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separator=" ")
    return splitter.split_text(transcripcion)


# Función para vectorizar los fragmentos de texto
def generar_vectorstore(api_key: str, docs: list) -> FAISS:
    os.environ["GOOGLE_API_KEY"] = api_key
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    docs_objects = [Document(page_content=chunk) for chunk in docs]
    vectorstore = FAISS.from_documents(docs_objects, embedding_model)
    return vectorstore


# Configurar modelo LLM
def configurar_llm(api_key: str) -> ChatGoogleGenerativeAI:
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.9)


# Configurar cadena de preguntas
def configurar_cadena_preguntas(llm, vectorstore, PROMPT_TEMPLATE):
    prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                            input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff", chain_type_kwargs={"prompt": prompt}
    )


# Función para hacer preguntas al modelo
def hacer_pregunta(qa_chain, consulta):
    try:
        respuesta = qa_chain.invoke({"query": consulta}).get(
            "result", "Error en la respuesta")
        return respuesta
    except Exception as e:
        return f"Error al procesar la consulta: {str(e)}"


# Interfaz en Streamlit
st.title("📹 Análisis Inteligente de Videos de YouTube 1")

# Input para ingresar la URL del video
url_video = st.text_input("Ingrese la URL del video")

# Inicializar estado de sesión
if "ultima_url" not in st.session_state:
    st.session_state.ultima_url = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


def procesar_video(url_video):
    if url_video == st.session_state.ultima_url:
        st.error("Por favor ingresa una url distinta")
        # Se pregunta si se esta procesando un video nuevo o el mismo
    if url_video != st.session_state.ultima_url or st.session_state.ultima_url is None:
        with st.status("⏳ Procesando...", expanded=True) as status:
            st.session_state.ultima_url = url_video

            # Obtiene la transcripcion
            transcripcion = obtener_subtitulos(url_video)
            st.session_state.transcripcion = transcripcion
            st.write("✅ Transcripción completada")

            # Fragmenta el texto
            docs = fragmentar_texto(transcripcion)
            st.session_state.docs = docs
            vectores = generar_vectorstore(
                "AIzaSyAMyslo9ANcFTQtXQ7l9Rtg7V4PaCghzr4", docs)
            st.session_state.vectores = vectores

            st.write("✅ Vectorización completada")

            # Configura el modelo LLM
            llm = configurar_llm("AIzaSyAMyslo9ANcFTQtXQ7l9Rtg7V4PaCghzr4")
            st.session_state.llm = llm

            st.write("✅ Conexión con el modelo LLM exitosa")

            PROMPT_TEMPLATE = """
            Responde en español (a menos de que en la pregunta se especifique otro idioma explícitamente) a la siguiente consulta basada en el contexto proporcionado.
            Si la consulta no está relacionada con el tema del video, responde con:
            "TEMA NO TRATADO EN EL VIDEO".
            Contexto:
            {context}
            Pregunta:
            {question}?
            """
            st.session_state.qa_chain = configurar_cadena_preguntas(
                llm, vectores, PROMPT_TEMPLATE)


def consultar_ia():
    """Permite hacer preguntas al modelo si está listo."""
    if st.session_state.qa_chain:
        pregunta = st.text_input("Haz una pregunta sobre el video:")
        if st.button("Consultar"):
            respuesta = hacer_pregunta(st.session_state.qa_chain, pregunta)
            st.write("### Respuesta:", respuesta)


# Botón para procesar el video
if st.button("Procesar Video"):

    if not url_video or url_video == "":
        st.error("Debes proporcionar una URL de video")

    else:
        if "youtube.com" in url_video or "youtu.be" in url_video:

            procesar_video(url_video)
        else:
            st.error("Por favor introduzca un video de la plataforma You Tube")


consultar_ia()
