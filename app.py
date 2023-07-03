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

def download_pdf():
    # Create a BytesIO object to store the PDF data
    buffer = BytesIO()
    
    # Generate or fetch your PDF data
    # Replace the following line with your code to generate or fetch the PDF data
    with open('constitution.pdf', 'rb') as file:
        pdf_data = file.read()
    
    # Write the PDF data to the buffer
    buffer.write(pdf_data)
    
    # Set the buffer's position back to the start
    buffer.seek(0)
    
    # Return the buffer
    return buffer


# Sidebar contents
with st.sidebar:
    st.title('Chat with PDF using LLM')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) GPT-3.5-Turbo

    ## This webapp was developed by Barry Rerecich
 
    ''')
    add_vertical_space(5)
    st.write('Example PDF: The Constitution Of United States')
    buffer = download_pdf()
    st.download_button(label='Download PDF', data=buffer, file_name='USA_Constitution.pdf', mime='application/pdf')


    
    st.markdown('''
    ## Example Questions:
    
    - What is the supreme law of the land in the United States?
    - Give me a chicken salad recipe please?
    - What does the Fourth Amendment say about "search and seizure"?
    - What is the role of the President according to the Constitution?
    - Which amendment guarantees the right to bear arms?
    ''')




def main():

    
    st.header('Chat With Your PDF')

    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 


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


        #Embedding using OpenAI
        store_name = pdf.name[:-4]

        st.write(f'{store_name}')
        
        #Store Embeddings Locally & Check Against That
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
        #Take In User Input
        query = st.text_input("Ask Questions Regarding Your File:")
        prompt = ("You are a helpful AI assistant, you are given context from an uploaded PDF, you are only to answer using the context given. Do not give any other information outside of the context given")
        
        

        #Grabs 3 Most Relevant Chunks from VectorStore
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            #Configuring LLM
            llm = ChatOpenAI(model_name='gpt-3.5-turbo')

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:

                response = chain.run(input_documents=docs, question=query + prompt)
                
            st.write(response)
            st.write(cb)

           

        
        







if __name__ == '__main__':
    main()
