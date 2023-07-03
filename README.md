[PDF Chatter App Link](https://barrypdfchatter.streamlit.app/)

# PDF-Based QA Chatbot with Streamlit and OpenAI

This application serves as a PDF reader that ingests a PDF, breaks it into text chunks, vector embeds each chunk, then leverages the GPT chat model to extract a few relevant chunks and includes them into the prompt as context to answer user questions.

## Table of Contents
- [Import Libraries]
- [Download and Display PDF]
- [Main Function]
  - [OpenAI API Key]
  - [PDF Upload and Processing]
  - [Text Embedding & Persistence]
  - [User Input and Question-Answering]

## Import Libraries

This section involves importing all necessary libraries to power the application, such as Python libraries for reading PDFs, creating a Streamlit web application, and specific modules for natural language processing tasks.

```python
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from io import BytesIO
```


## Download and Display PDF
This section sets up the sidebar with an application title, a brief description, an example PDF, and some example questions that users can ask the application. Also, the function to download the PDF file is defined.

```python
# Sidebar contents
with st.sidebar:
    st.title('Chat with PDF using LLM')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) GPT-3.5-Turbo / Embeddings

    ## This webapp was developed by Barry Rerecich
 
    ''')
    add_vertical_space(5)
    st.write('Example PDF: The Constitution Of United States')
    buffer = download_pdf()
    st.download_button(label='Download PDF', data=buffer, file_name='USA_Constitution.pdf', mime='application/pdf')
```


## Main Function
The main function is where the application logic lives. It consists of several subsections:


## OpenAI API Key
This section sets up the OpenAI API key, which is required to make requests to OpenAI's servers.
```python
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```


## PDF Upload and Processing
This section allows the user to upload a PDF file, which is then processed and split into text chunks using Langchain's RecursiveCharacterTextSplitter.

```python
#Upload PDF File
pdf = st.file_uploader('Upload Your PDF', type='pdf')

#Langchain Chunk Splitter
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)
```



## Text Embedding & Persistence
The chunks of text are then vector embedded using OpenAI Embeddings and stored locally for later retrieval and comparison.
```python
#Embedding using OpenAI
store_name = pdf.name[:-4]

if os.path.exists(f"{store_name}.pkl"):
    with open(f"{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
        
else:
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)
```



## User Input and Question-Answering
This final section allows the user to ask questions. The questions are used to retrieve the most relevant chunks from the VectorStore. The selected chunks are then fed into the GPT-3.5-turbo model to generate answers.

```python
#Take In User Input
query = st.text_input("Ask Questions Regarding Your File:")
prompt = ("You are a helpful AI assistant, you are given context from an uploaded PDF, you are only to answer using the context given. Do not give any other information outside of the context given")
```

After receiving the user's query, the system executes a similarity search over the previously embedded text chunks. The top three most relevant chunks are retrieved for further processing.


```python
#Grabs 3 Most Relevant Chunks from VectorStore
if query:
    docs = VectorStore.similarity_search(query=query, k=3)
```

With these relevant chunks, the application now configures the GPT-3.5-turbo model and sets up a Question-Answering (QA) chain. The model is given both the relevant text chunks (as input_documents) and the user's query (plus the standard prompt), and then generates a response.


```python
#Configuring LLM
llm = ChatOpenAI(model_name='gpt-3.5-turbo')

chain = load_qa_chain(llm=llm, chain_type="stuff")
with get_openai_callback() as cb:
    response = chain.run(input_documents=docs, question=query + prompt)
                
st.write(response)
st.write(cb)
```

This response is then displayed on the Streamlit interface for the user to view. The entire process can then be repeated for new queries or documents.

