import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import FAISS
from langchain.schema import Document  # Usando o schema correto de langchain
from langchain_ollama import ChatOllama
import pandas as pd
# Carrega variáveis de ambiente e chaves de acesso.
_ = load_dotenv(find_dotenv())

# Configuração do servidor Ollama
ollama_server_url = "http://localhost:11434" 
model_local = ChatOllama(model="llama3.1:8b-instruct-q4_K_S")

@st.cache_data
def load_csv_data():    
    # Link direto para o arquivo CSV no GitHub
    github_csv_url = "https://raw.githubusercontent.com/HelbertMoura/ollama/refs/heads/main/knowledge_base.csv"
                       

    # Carregar dados do CSV do GitHub
    df = pd.read_csv(github_csv_url)

    # Exibir alguns dados para verificar o carregamento
    st.write("Dados carregados:", df.head())

    # Criar documentos com o atributo `page_content`
    documents = [Document(page_content=row['pergunta'], metadata={"resposta": row['resposta']}) for _, row in df.iterrows()]

    # No mesmo servidor, uso também um modelo de Embedding
    embeddings = OllamaEmbeddings(base_url=ollama_server_url,
                                  model='nomic-embed-text')

    # Criar o vectorstore com FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Chamar a função de carregar os dados
retriever = load_csv_data()

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
