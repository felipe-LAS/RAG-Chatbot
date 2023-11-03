import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from tqdm.auto import tqdm  
from utils import *


def init():
    # carrega a chave API
    load_dotenv()
    
    # testa que a chave API existe
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # configura pagina streamlit
    st.set_page_config(
        page_title="Dúvidas do vestibular da Unicamp",
        page_icon="🤖"
    )

def main():
    init()

    chat = ChatOpenAI(temperature=0)

    # historico de mensagem
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Dúvidas - Vestibular Unicamp")

    # sidebar com user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")

        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Procurando..."):
                pinecone_results = search_pinecone(user_input)
                response_content = process_pinecone_results(pinecone_results)
                response = AIMessage(content=response_content)
                st.session_state.messages.append(response)

    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


# Inicializar Pinecone
pinecone.init(
    api_key='8d7ba2d0-2c05-47d7-bec7-b1d6bc9d03c7',
    environment='gcp-starter'
)

text_field = "text"  

# Nome e inicialização do índice
index_name = 'final-chatbot'
index = pinecone.Index(index_name)
index.describe_index_stats()

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Caminho para o arquivo txt
file_path = '/Users/felipesantos/Desktop/neuralmind/Procuradoria-Geral-Normas.txt'

# Ler o arquivo TXT
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

# Dividir o texto em linhas
lines = data.split('\n')

# Tamanho do lote para processamento
batch_size = 100

# Processar e indexar os dados em lotes
for i in tqdm(range(0, len(lines), batch_size)):
    i_end = min(len(lines), i+batch_size)
    # Obter lote de dados
    batch = lines[i:i_end]
    # Gerar IDs únicos para cada linha
    ids = [f'{i}-{j}' for j in range(i, i_end)]
    # Obter texto para embedding
    texts = batch
    # Embedding do texto
    embeds = embed_model.embed_documents(texts)
    # Obter metadados para armazenar no Pinecone
    metadata = [{'text': text} for text in batch]
    # Adicionar ao Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

vectorstore = pinecone(
    index, embed_model.embed_query, text_field
)

def search_pinecone(query):
    results = vectorstore.similarity_search(query, k=3)  # busca os 3 documentos mais similares
    return results

def process_pinecone_results(results):
    response_content = ""
    for result in results:
        id, score, metadata = result  # assumindo que o resultado contém id, score e metadata
        response_content += f"Resultado: {metadata['text']}, Score: {score}\n"
    return response_content

if __name__ == '__main__':
    main()