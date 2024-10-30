import mysql.connector
import requests
import streamlit as st
import json
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Conectar ao banco de dados MySQL da clínica médica
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="21399150Helb@",
    database="VOX"  # Nome do banco de dados da clínica
)

# Inicializar o cursor
cursor = conn.cursor()

# Estrutura estática do banco de dados
estrutura_banco_fixa = {
    "paciente": ["id", "nome", "idade", "genero"],
}

# Carregar o ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Função para enviar perguntas clínicas usando ClinicalBERT
def enviar_pergunta_clinicalbert(pergunta):
    inputs = tokenizer(pergunta, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    
    return f"Resposta clínica baseada em ClinicalBERT: Classe {predicted_class}"

# Função para gerar consultas SQL
def gerar_consulta_sql(pergunta):
    # Código para gerar SQL (mesmo que anteriormente)
    pass

# Função para enviar perguntas médicas
def enviar_pergunta_medicina(pergunta):
    resposta = enviar_pergunta_clinicalbert(pergunta)
    return resposta

# Função para executar a consulta SQL no banco de dados MySQL
def executar_consulta_sql(consulta_sql):
    # Código para executar a consulta SQL (mesmo que anteriormente)
    pass

# Interface do chatbot no Streamlit
st.title("Chatbot para Clínica Médica (usando ClinicalBERT)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Digite sua pergunta:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Identificar se a pergunta é médica ou SQL
    if "paciente" in user_input.lower() or "banco" in user_input.lower() or "sql" in user_input.lower():
        consulta_sql = gerar_consulta_sql(user_input)
        st.write(f"Consulta SQL gerada: {consulta_sql}")
        resposta = executar_consulta_sql(consulta_sql)
    else:
        resposta = enviar_pergunta_medicina(user_input)

    # Exibir a resposta
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    with st.chat_message("assistant"):
        st.markdown(resposta)
