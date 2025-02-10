"""This is a streamlit app to do research"""

import subprocess
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
import os
import re
import psutil
import logging



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def pdf_read(pdf_doc):
    """Read the text from PDF document."""
    logging.debug("Reading PDF document")
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    logging.debug("Finished reading PDF document")
    return text

def create_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Create text chunks from a large text block."""
    logging.debug("Creating text chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_text(text)
    logging.debug(f"Created {len(text_chunks)} text chunks")
    return text_chunks

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def vector_store(text_chunks):
    """Create a vector store for the text chunks."""
    logging.debug("Creating vector store")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")
    logging.debug("Vector store created and saved locally")

def query_llama_via_cli(input_text):
    """Query the Llama model via the CLI."""
    logging.debug("Querying Llama model via CLI")
    try:
        process = subprocess.Popen(
            ["ollama", "run", "deepseek-coder"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1
        )
        stdout, stderr = process.communicate(input=f"{input_text}\n", timeout=30)
        if process.returncode != 0:
            logging.error(f"Error in the model request: {stderr.strip()}")
            return f"Error in the model request: {stderr.strip()}"
        response = re.sub(r'\x1b\[.*?m', '', stdout)
        return extract_relevant_answer(response)
    except subprocess.TimeoutExpired:
        process.kill()
        logging.error("Timeout for the model request")
        return "Timeout for the model request"
    except Exception as e:
        logging.error(f"An unexpected error has occurred: {str(e)}")
        return f"An unexpected error has occurred: {str(e)}"

def extract_relevant_answer(full_response):
    """Extract the relevant response from the full model response."""
    logging.debug("Extracting relevant answer from model response")
    response_lines = full_response.splitlines()
    if response_lines:
        return "\n".join(response_lines).strip()
    return "No answer received"

def get_conversational_chain(context, ques):
    """Create the input for the model based on the prompt and context."""
    logging.debug("Creating conversational chain")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent and helpful assistant. Your goal is to provide the most accurate and detailed answers 
                possible to any question you receive. Use all available context to enhance your answers, and explain complex 
                concepts in a simple manner. If additional information might help, suggest further areas for exploration. If the 
                answer is not available in the provided context, state this clearly and offer related insights when possible.""",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    input_text = f"Prompt: {prompt.format(input=ques)}\nContext: {context}\nQuestion: {ques}"
    response = query_llama_via_cli(input_text)
    st.write("PDF: ", response)

def user_input(user_question, pdf_text):
    """Processes the user input and calls up the model."""
    logging.debug("Processing user input")
    context = pdf_text
    get_conversational_chain(context, user_question)

def main():
    """Main function of the Streamlit application."""
    logging.debug("Starting Streamlit application")
    st.set_page_config(page_title="CHAT WITH YOUR PDF")
    st.header("PDF CHAT APP")

    pdf_text = ""
    pdf_doc = st.file_uploader("Upload your PDF Files and confirm your question", accept_multiple_files=True)

    if pdf_doc:
        pdf_text = pdf_read(pdf_doc)

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and pdf_text:
        user_input(user_question, pdf_text)

    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)
    st.sidebar.write(f"Memory usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    main()
