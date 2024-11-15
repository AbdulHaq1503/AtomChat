Overview
-------------------------------------------------------------
AtomChat 1.0 and AtomChat2.0 are Streamlit-based applications that allow users to interact with PDF documents using a conversational interface. Both applications are based off of locally installed Llama3.1.

AtomChat1.0
-------------------------------------------------------------
This is a more specific chat-bot that answers queries based on default documents provided to the model.

AtomChat2.0
-------------------------------------------------------------
This is a generalized chat-bot that uses documents provided by the user apart from the default documents that it has already. In case the user doesnt provide any documents the bot can use the default documents to answer the user queries.

General Information:
-------------------------------------------------------------

 1. Used locally installed Llama 3.1 as the base llm
 2. Used all-MiniLM-L6-v2 from HuggingFace for embeddings
 3. Used FAISS database to store and retrieve document chunks efficiently
 4. Utilized LangChain to streamline the integration of large language models (LLMs), embedding models, and memory for conversational capabilities.
 5. Used Streamlit for the user-interface design

Installation
-------------------------------------------------------------
1. Ensure you have Llama 3.1 installed on your local system
2. Clone the Repository using:
    ``` 
        git clone <repository-url>
        cd AtomChat
    ```
3. Install python version mentioned in the python_version document
4. Install the required dependencies:  
    Note: Ensure you have pip and a Python environment (e.g., virtualenv or Conda) set up before running this command.
    ```
        pip install -r requirements.txt
    ```
5. Run the application:
    ```
    streamlit run atomchat1.py
    streamlit run atomchat2.py
    ```

How It Works
-------------------------------------------------------------

1. PDF Text Extraction: Extracts text from the specified PDF documents using PyPDF2.

2. Text Chunking: Splits the extracted text into smaller chunks (700 characters with 200-character overlap)using  LangChain's CharacterTextSplitter.

3. Embedding and Storage: Creates embeddings using the Hugging Face all-MiniLM-L6-v2 model via LangChain.
Stores embeddings in a FAISS vector database for efficient similarity-based retrieval.
4. Conversational Retrieval: Uses LangChain's ConversationalRetrievalChain to connect the LLM (ChatOllama) with the vector store. Maintains conversational context using ConversationBufferMemory.
5. Streamlit Interface: Users input queries into the chat interface.Responses are generated in real-time, displayed in a chat layout.

Customizations
-------------------------------------------------------------
1. Defaunt Documents: Add your default documents according to your need by modifying the doc_paths list.
2. Embedding Model: Replace all-MiniLM-L6-v2 with other Hugging Face models supported by LangChain.
3. LLM: Swap ChatOllama with any LangChain-compatible LLM for customization.
