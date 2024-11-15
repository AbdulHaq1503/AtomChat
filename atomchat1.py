import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader

from langchain_community.chat_models.ollama import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings


#llm = Llama(model_path=r"C:\Users\MOHAMMED ABDUL HAQ\.ollama\models\blobs\sha256-8eeb52dfb3bb9aefdf9d1ef24b3bdbcfbe82238798c4b918278320b6fcef18fe")

# function for extracting text from all the documents into one string
def get_doc_txt(doc_paths): 
    doc_text = ""
    for doc in doc_paths:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            doc_text += page.extract_text()
    return doc_text


# function for Dividing text into chunks
def get_chunker(txt_var):
    splitter = CharacterTextSplitter(separator = '\n', chunk_size = 700, chunk_overlap = 200, length_function = len )
    chunks = splitter.split_text(txt_var)
    return chunks


# function to create embeddings and store in faiss vector-db
def get_vectorstore(chunks):
    #embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts = chunks, embedding = embeddings)
    return vectorstore


def get_conversationChain(vector_store):
    llm = ChatOllama(model="llama3.1", temperature=0)
    memory = ConversationBufferMemory(
        memory_key = 'chat_history',
        return_messages = True      
        )#stores conversation history in a structured way
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain


def answer_input(input_question):
    response = st.session_state.conversation({'question':input_question})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            with st.chat_message("human"):
                st.markdown(message.content)
        else:
            with st.chat_message("ai"):
                st.markdown(message.content)
    


def main():
    
    doc_paths = [r'C:\Users\MOHAMMED ABDUL HAQ\Desktop\goog-10-k-2023 (1).pdf',
                  r'C:\Users\MOHAMMED ABDUL HAQ\Desktop\tsla-20231231-gen.pdf',
                    r'C:\Users\MOHAMMED ABDUL HAQ\Desktop\uber-10-k-2023.pdf']
    
    st.set_page_config(page_title = "AtomChat")
    st.header("AtomChat1.0: Chat with PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] 
    
    if st.session_state.conversation is None:
        with st.spinner("*Initializing*....."):
            # extracted text
            raw_text = get_doc_txt(doc_paths)

            # breaking text into chunks
            text_chunks = get_chunker(raw_text)

            #storing vectors in faiss-db
            vectorstore = get_vectorstore(text_chunks)

            #creating conversation chain
            st.session_state.conversation = get_conversationChain(vectorstore)

    query = st.chat_input("Enter your query here")
    if query:
        with st.spinner("*Thinking*......"):
            answer_input(query)

if __name__ == '__main__':
    main()