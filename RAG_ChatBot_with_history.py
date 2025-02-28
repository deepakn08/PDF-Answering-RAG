import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

import streamlit as st

from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload PDFs and chat with it")

api_key = st.text_input("Enter your Groq API key", type = 'password')

model_input = st.selectbox(
    'Select a model',
     ['gemma2-9b-it', 'llama3-8b-8192', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'llama-3.3-70b-versatile', 'llama3-70b-8192'])

if api_key:
    st.write('Selected model is:', model_input)
    llm = ChatGroq(model = model_input, api_key = api_key)
    session_id = st.text_input("Session_id", value = 'Default1')
    
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            tempPdf = f'./temp.pdf'
            with open(tempPdf, 'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
                
            loader = PyPDFLoader(tempPdf)
            docs = loader.load()
            documents.extend(docs)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)
        vectorstore = FAISS.from_documents(documents = splits, embedding = embeddings)
        retriever = vectorstore.as_retriever()

        
        contextualized_system_prompt = """Given a chat history and the latest user question/n
        which might reference context in the chat history,/n
        formulate a standalone question which can be understood/n
        without the chat history. Do not answer the question,/n
        just reformulate it if needed and otherwise return it as it is."""

        contextualized_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualized_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_prompt)
    
        qa_system_prompt = """You are an assistant for question-answering tasks./n
        Use the following pieces of retrieved context to answer/n
        the question. If the question is from outside the input PDF then tell 'Sorry, I have no idea'./n
        Use three statements maximum and keep the answer concise/n/n
        <context>
        
        {context}
        
        </context>
        """
    
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', qa_system_prompt),
                MessagesPlaceholder(    "chat_history"),
                ('human', '{input}')
            ]
        )
        
        question_answer_doc_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_doc_chain)
        
        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
                
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(rag_chain, 
                                        get_session_history,
                                        input_messages_key='input',
                                        history_messages_key='chat_history',
                                        output_messages_key='answer'
                                    )
        
        user_input = st.text_input("Your question:")
        
        if user_input:
            response = conversational_rag_chain.invoke({'input': user_input},
                                                config = {'configurable': {'session_id':session_id}})
            
            st.write("Assistant:", response['answer'])
            
else:
    st.warning("Please enter your Groq API key")
