from flask import Flask, request, jsonify, render_template
import requests
import os
import json
import mysql.connector
from mysql.connector import Error
import re
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Definir o caminho para o diretório de templates
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# URL do servidor Ollama (certifique-se de que o Ollama está em execução)
OLLAMA_SERVER_URL = "http://localhost:11434/api/generate"

# Função para conectar ao MySQL
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='VOX',
            user='root',
            password='21399150Helb@'
        )
        if connection.is_connected():
            print("Conexão com o MySQL foi bem-sucedida")
        return connection
    except Error as e:
        print(f"Erro ao conectar com o MySQL: {e}")
        return None

# Função para realizar uma consulta no banco de dados
def consultar_dados(query):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            print(f"Executando consulta: {query.strip()}")
            cursor.execute(query.strip())
            result = cursor.fetchall()
            print(f"Consulta executada com sucesso. Resultados obtidos: {result}")
        except Error as e:
            print(f"Erro ao consultar o banco de dados: {e}")
            result = None
        finally:
            cursor.close()
            connection.close()
            print("Conexão com o banco de dados fechada.")
        return result
    else:
        print("Conexão com o banco de dados não foi estabelecida.")
        return None

# Função para validar consulta SQL
def validar_consulta_sql(query):
    # Verificar se a consulta contém as palavras-chave esperadas e não tem comandos perigosos
    if "SELECT" in query.upper() and "FROM" in query.upper() and query.count("'") % 2 == 0 and ";" in query:
        return True
    return False

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message").lower()
    user_message = re.sub(r'[^\w\s]', '', user_message)  # Limpar a entrada do usuário

    # Definir se a pergunta deve ser respondida pelo banco de dados ou diretamente pela IA
    if any(keyword in user_message for keyword in ["paciente", "profissional", "atendimento", "banco", "sql", "consulta"]):
        # Pergunta relacionada ao banco de dados
        prompt = f"""
        O banco de dados contém as seguintes tabelas principais:

        - Tabela 'profissional':
        - Colunas: idProfissional (int), nome (varchar), cpf (varchar), conselhoProfissional (varchar), status (smallint)

        - Tabela 'paciente':
        - Colunas: idPaciente (int), nome (varchar), cpf (varchar), sexo (smallint), status (smallint)

        - Tabela 'atendimento':
        - Colunas: idAtendimento (int), idPaciente (int), idProfissional (int), dtExecucao (datetime), dataAtendimento (date), status (smallint)

        Instruções:
        1. Gere uma **consulta SQL** baseada na pergunta do usuário.
        2. Use as tabelas **'atendimento'** para consultas e **'paciente'** para informações dos pacientes.
        3. Para filtrar consultas por data, use a coluna **dataAtendimento** na tabela **atendimento**.
        4. Retorne a **consulta SQL** diretamente, sem explicações adicionais.

        Pergunta do usuário: {user_message}
        """
    else:
        # Pergunta genérica não relacionada ao banco de dados
        prompt = f"Pergunta do usuário: {user_message}\nPor favor, forneça uma resposta direta à pergunta acima."

    try:
        payload = {
            "model": "llama3.1:8b-instruct-q4_K_S",
            "prompt": prompt
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(OLLAMA_SERVER_URL, json=payload, headers=headers, stream=True)
        response.raise_for_status()

        # Coletar a resposta incremental do servidor Ollama
        full_response = []
        for line in response.iter_lines():
            if line:
                try:
                    json_line = line.decode('utf-8')
                    parsed_line = json.loads(json_line)
                    full_response.append(parsed_line.get("response", ""))
                    if parsed_line.get("done", False):
                        break
                except json.JSONDecodeError as e:
                    print(f"Erro ao decodificar JSON: {e}")
                    continue
        full_response = "".join(full_response)

        if any(keyword in user_message for keyword in ["paciente", "profissional", "atendimento"]):
            consulta_sql = re.sub(r'```[a-zA-Z]*', '', full_response.strip())
            consulta_sql = consulta_sql.strip().rstrip(";")  # Remover espaços e pontuações desnecessárias e garantir que não termine com ponto e vírgula

            # Garantir que a consulta tenha aspas fechadas
            if consulta_sql.count("'") % 2 != 0:
                consulta_sql += "'"  # Adicionar uma aspa simples se estiver faltando

            # Substituir '=' por 'LIKE' para tornar a consulta mais flexível
            consulta_sql = re.sub(r"= '([^']*)'", r"LIKE '%\1%'", consulta_sql)

            consulta_sql += ';'  # Garantir que a consulta termine com ponto e vírgula

            print(f"Consulta SQL gerada pela IA: {consulta_sql}")

            # Validar consulta antes de executar
            if not validar_consulta_sql(consulta_sql):
                raise ValueError("Consulta SQL inválida.")

            # Executar a consulta SQL gerada
            resultado_banco = consultar_dados(consulta_sql)
            if resultado_banco:
                # Melhorar a apresentação dos resultados em HTML com nome e CPF
                resposta = "<div class='result-container'><h3>Pacientes atendidos:</h3><ul class='patient-list'>"
                for row in resultado_banco:
                    nome = row[1] if row[1] and row[1] != '\x00' else "Nome não informado"  # Garantir que o nome do paciente seja exibido ou "Nome não informado" caso não exista
                    resposta += f"<li class='patient-item'><span class='patient-icon'>✔️</span> {nome}</li>"
                resposta += "</ul></div>"

                # Gerar gráfico de atendimentos se houver dados suficientes
                if len(resultado_banco) > 1:
                    labels = [row[1] if row[1] and row[1] != '\x00' else "Nome não informado" for row in resultado_banco]
                    values = [1 for _ in resultado_banco]
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(labels)), values, tick_label=labels, color='skyblue')
                    plt.xlabel('Pacientes')
                    plt.ylabel('Número de Atendimentos')
                    plt.title('Número de Atendimentos por Paciente')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    graph_url = base64.b64encode(buf.getvalue()).decode()
                    plt.close()
                    resposta += f"<div class='chart-container' style='text-align: center; margin-top: 20px;'><h3>Gráfico de Atendimentos:</h3><img src='data:image/png;base64,{graph_url}' alt='Gráfico de Atendimentos' style='max-width: 100%; height: auto;'></div>"
            else:
                resposta = "<p>A consulta foi executada, mas não retornou resultados. Verifique se há registros na data solicitada.</p>"
        else:
            resposta = f"<p>{full_response}</p>"

    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Erro ao processar solicitação: {e}")
        resposta = "<p>Houve um problema ao processar sua pergunta. Tente novamente mais tarde.</p>"

    # Log final para revisar a resposta
    print(f"Resposta enviada ao usuário: {resposta}")

    return jsonify({"response": resposta})

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=8000, debug=True)
