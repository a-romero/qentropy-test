import os
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

from llama_index import LangchainEmbedding, ServiceContext, StorageContext, download_loader, LLMPredictor
from llama_index.vector_stores import ChromaVectorStore
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, load_index_from_storage
from llama_index.storage.index_store import SimpleIndexStore

import chromadb
from chromadb.config import Settings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def process_docs(pdf_docs, project):
    input_files = []
    for pdf in pdf_docs:
        input_files.append('./data/'+project+'/'+pdf.name)
    return input_files

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def store_vector(input_files, chroma_client, project, llm):
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings())
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    chroma_collection = chroma_client.create_collection(project) #create chroma collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    ## init service context
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm,
        embed_model=embeddings
    )
    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    index = GPTVectorStoreIndex.from_documents(documents=documents,
                                                        storage_context=storage_context,
                                                        service_context=service_context)
    ## save index
    index.set_index_id(project)
    index.storage_context.persist('./storage/index_storage/'+project+'/')
    return vector_store, embeddings

def conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    #load_storage_context = StorageContext.from_defaults(
    #    vector_store=vector_store,
    #    index_store=SimpleIndexStore.from_persist_dir(persist_dir='./storage/index_storage/'+project+'/'),
    #)
    #load_service_context = ServiceContext.from_defaults(llm_predictor=llm,embed_model=embeddings)
    #load_index = load_index_from_storage(service_context=load_service_context, 
    #                            storage_context=load_storage_context)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store,
        memory=memory
    )
    #query = load_index.as_query_engine()
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def get_projects():
    projects = {}
    for directory in os.listdir("./data/"):
        if os.path.isdir(os.path.join("./data/", directory)):
            projects[directory] = None
    return projects

def main():
    load_dotenv()
    img = Image.open('./resources/qentropy_logo_black.png')
    st.set_page_config(page_title="Astrolabe Insights",
                       page_icon=img)
    st.write(css, unsafe_allow_html=True)

    #llm = ChatOpenAI()
    llm = LLMPredictor(llm=ChatOpenAI(temperature=0.2, max_tokens=512, model_name='gpt-4'))
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    chroma_client = chromadb.Client(
          Settings(chroma_db_impl="duckdb+parquet",
           persist_directory="./storage/vector_storage/chromadb/"
    ))

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.image(img, width=80)
    st.header("Astrolabe Insights")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Projects")
        projects = get_projects()
        existing_project = st.sidebar.selectbox("Select a project", projects.keys())
        #if st.button("Load project"):
        #    project = existing_project
        project = existing_project
        #project = st.sidebar.text_input('Create a new project:')
        #if project not in projects:
        #    path = os.path.join("./storage/index_storage/", project)
        #    os.mkdir(path)
        #    projects['project'] = None

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                #raw_text = get_pdf_text(pdf_docs)
                input_files = process_docs(pdf_docs, project)

                # get the text chunks
                #text_chunks = get_text_chunks(raw_text)

                # create vector store
                vector_store, embeddings = store_vector(input_files, chroma_client, project, llm)

                # create conversation chain
                st.session_state.conversation = conversation_chain(
                    vector_store)


if __name__ == '__main__':
    main()