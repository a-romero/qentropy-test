
import os
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader
import app_utils, htmlTemplates, widgets

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, project):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local(project)
    return vectorstore


def get_conversation_chain(project):
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    vectorstore = FAISS.load_local(project, embeddings=embeddings)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
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
    img_full_logo = Image.open('./resources/qentropy_full_logo.png')
    img_astrolabe_logo = Image.open('./resources/AstrolabeInsights_black.png')
    img = Image.open('./resources/qentropy_logo_black.png')
    st.set_page_config(page_title="Astrolabe Insights",
                       page_icon=img, layout='wide')
    st.write(css, unsafe_allow_html=True)

    # Initialise session values
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    if 'projects' not in st.session_state:
        st.session_state.projects = []
    if 'project_info' not in st.session_state:
        st.session_state.project_info = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'download' not in st.session_state:
        st.session_state.download = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0


    with st.sidebar:
        st.image(img_full_logo, width=200, output_format='PNG')
        st.sidebar.title("Projects")
        project_holder = st.empty()
        widgets.add_project()
        widgets.delete_project()
        
        if len(st.session_state.projects) == 0:
            project_holder.write('No available project. Please add a new project.')
        else:
            try:
                i = st.session_state.projects.index(st.session_state.current_project)
            except ValueError:
                i = 0

            # ignore index if switching project (change radio button)
            if st.session_state.get('switching_project', False):
                current_project = project_holder.radio(
                    'Select a project to work with:', st.session_state.projects,
                    on_change=lambda: st.session_state.update({'switching_project': True})
                )  # set switching project state to apply the changes
                st.session_state.pop('switching_project')
            else:
                current_project = project_holder.radio(
                    'Select a project to work with:', st.session_state.projects, i,
                    on_change=lambda: st.session_state.update({'switching_project': True})
                )  # set index to persist with the selected project during pagination
            st.session_state.current_project = current_project
        
        #projects = get_projects()
        #existing_project = st.sidebar.selectbox("Select a project", projects.keys())
        #project = existing_project

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks, current_project)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    current_project)

    buffer, left_column, _, right_column = st.columns([1, 15, 1, 4])
    #left_column, inter_col_space1, graph_area, inter_col_space2, _, inter_col_space3, right_column = st.columns((3, 0.05, 2, 0.05, 2, 0.05, 2))
    # display and update project info at the right column
    if st.session_state.current_project is not None:
        # get project info for the first time or when switching projects
        if st.session_state.project_info is None or \
                st.session_state.project_info['project'] != st.session_state.current_project:
            app_utils.get_project_info()

        with right_column:                
            # display project name
            st.header(st.session_state.current_project)
            # display project creation datetime
            st.write(htmlTemplates.create_date_html(st.session_state.project_info['createDate']),
                     unsafe_allow_html=True)
            # project description text area
            widgets.project_description()
            # placeholder to display the labelling progress
            progress_holder = st.empty()
            # placeholder to display list of labels
            label_list_holder = st.empty()
            # expander to add label
            widgets.add_label()
            # expander to delete label
            widgets.delete_label()
    
    with left_column:
        st.image(img_astrolabe_logo, width=300, output_format='PNG')
        user_question = st.text_input("Ask a question about your documents:", key="chat_input")
        if user_question:
            handle_userinput(user_question)
        if st.session_state.conversation is not None:
                nodes = []
                edges = []
                nodes.append( Node(id="Humn.ai", 
                                label="Humn.ai Ltd", 
                                size=25, 
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/iew0h2sbbx40fy3jr3vm") 
                            ) # includes **kwargs
                nodes.append( Node(id="Walsingham",
                                label="Walsingham Motor Insurance Limited",
                                size=25,
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/v1480506759/eqbmarbhk2vw0soxg8yw.png") 
                            )
                nodes.append( Node(id="InsurtechGateway",
                                label="Insurtech Gateway",
                                size=25,
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/almsgfhel0en4al8opdz") 
                            )
                nodes.append( Node(id="ShellVentures",
                                label="Shell Ventures",
                                size=25,
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/v1488040409/mkz17efk4lbzqwtxoef8.png") 
                            )
                nodes.append( Node(id="Marbruck",
                                label="Marbruck",
                                size=25,
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/hiksnbv601utpulr8bnn") 
                            )
                nodes.append( Node(id="BXRGroup",
                                label="BXR Group",
                                size=25,
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/wmxiexhdafyrm6gjewkp") 
                            )
                nodes.append( Node(id="Wayra",
                                label="Wayra UK ScaleUpHub",
                                size=25,
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/wrxjxm1rlzg2tn2szdau") 
                            )
                nodes.append( Node(id="Innogy",
                                label="Innogy Innovation Hub",
                                size=25,
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/kvyqi0d0oqch3lnqnw0a") 
                            )
                nodes.append( Node(id="Telefonica",
                                label="Telefonica",
                                size=25,
                                shape="circularImage",
                                image="https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/kfedwiwwhttnk2nc183p") 
                            )
                edges.append( Edge(source="Humn.ai", 
                                label="purchased", 
                                target="Walsingham", 
                                # **kwargs
                                ) 
                            )
                edges.append( Edge(source="Humn.ai", 
                                label="received_funding_from", 
                                target="BXRGroup", 
                                # **kwargs
                                ) 
                            )
                edges.append( Edge(source="Humn.ai", 
                                label="received_funding_from", 
                                target="ShellVentures", 
                                # **kwargs
                                ) 
                            )
                edges.append( Edge(source="Humn.ai", 
                                label="received_funding_from", 
                                target="Marbruck", 
                                # **kwargs
                                ) 
                            )
                edges.append( Edge(source="Humn.ai", 
                                label="received_funding_from", 
                                target="InsurtechGateway", 
                                # **kwargs
                                ) 
                            )
                edges.append( Edge(source="Humn.ai", 
                                label="selected_for", 
                                target="Wayra", 
                                # **kwargs
                                ) 
                            )
                edges.append( Edge(source="Humn.ai", 
                                label="selected_for", 
                                target="Innogy", 
                                # **kwargs
                                ) 
                            )
                edges.append( Edge(source="Telefonica", 
                                label="partnered_with", 
                                target="Innogy", 
                                # **kwargs
                                ) 
                            )
                edges.append( Edge(source="Telefonica", 
                                label="owns", 
                                target="Wayra", 
                                # **kwargs
                                ) 
                            )

                config = Config(width=750,
                                height=950,
                                directed=True, 
                                physics=True, 
                                hierarchical=False,
                                # **kwargs
                                )

                agraph(nodes=nodes, 
                        edges=edges, 
                        config=config)
        

if __name__ == '__main__':
    main()