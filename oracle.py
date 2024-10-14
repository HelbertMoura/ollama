import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import FAISS
from langchain.schema import Document  # Usando o schema correto de langchain
from langchain_ollama import ChatOllama
import pandas as pd
import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
# Carrega variáveis de ambiente e chaves de acesso.
_ = load_dotenv(find_dotenv())

# Configuração do servidor Ollama
ollama_server_url = "http://localhost:11434" 
model_local = ChatOllama(model="llama3.1:8b-instruct-q4_K_S")

@st.cache_data
def load_csv_data():
    github_csv_url = "https://raw.githubusercontent.com/HelbertMoura/ollama/main/knowledge_base.csv"
    try:
        df = pd.read_csv(github_csv_url)
        st.write("Dados carregados:", df.head())

        # Verifica se as colunas 'pergunta' e 'resposta' existem
        if 'pergunta' not in df.columns or 'resposta' not in df.columns:
            st.error("O CSV deve conter as colunas 'pergunta' e 'resposta'.")
            return None
        
        # Cria os documentos, ignorando linhas com valores ausentes
        documents = []
        for _, row in df.iterrows():
            pergunta = row.get('pergunta')
            resposta = row.get('resposta')
            if pd.notna(pergunta) and pd.notna(resposta):
                documents.append(Document(page_content=pergunta, metadata={"resposta": resposta}))

        # Certifique-se de que o Ollama está funcionando e que o modelo existe
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model='nomic-embed-text')
        
        # Criar o vectorstore com FAISS
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return None

# Chamar a função para carregar os dados
retriever = load_csv_data()
if retriever is not None:
    st.write("Retriever carregado com sucesso.")

# Restante do código para o chatbot
st.title("Oráculo - Asimov Academy")

# Configuração do prompt e do modelo
rag_template = """
Você é um atendente de uma empresa.
Seu trabalho é conversar com os clientes, consultando a base de 
conhecimentos da empresa, e dar 
uma resposta simples e precisa para ele, baseada na 
base de dados da empresa fornecida como 
contexto.

Contexto: {context}

Pergunta do cliente: {question}
"""
human = "{text}"
prompt = ChatPromptTemplate.from_template(rag_template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de entrada para o usuário
if user_input := st.chat_input("Você:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Adiciona um container para a resposta do modelo
    response_stream = chain.stream({"text": user_input})    
    full_response = ""
    
    response_container = st.chat_message("assistant")
    response_text = response_container.empty()
    
    for partial_response in response_stream:
        full_response += str(partial_response.content)
        response_text.markdown(full_response + "▌")

    # Salva a resposta completa no histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response})
