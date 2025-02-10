"""This is an api way to do the pdf reading."""
import subprocess
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
import os
import re
import logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PdfQuestionAnswer:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        self.pdf_text = None
        self.vector_store = None
        self.embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
        self._process_pdf()

    def _process_pdf(self):
        """Reads PDF and creates text chunks and vector store."""
        logging.debug("Starting PDF processing")
        with open(self.pdf_file_path, "rb") as f:
            self.pdf_text = self.pdf_read([f])
        text_chunks = self.create_text_chunks(self.pdf_text)
        self.vector_store = self.vector_store_from_chunks(text_chunks)

    def pdf_read(self, pdf_doc):
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

    def create_text_chunks(self, text, chunk_size=1000, chunk_overlap=200):
        """Create text chunks from a large text block."""
        logging.debug("Creating text chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        text_chunks = text_splitter.split_text(text)
        logging.debug(f"Created {len(text_chunks)} text chunks")
        return text_chunks

    def vector_store_from_chunks(self, text_chunks):
        """Create a vector store for the text chunks."""
        logging.debug("Creating vector store")
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local("faiss_db")
        logging.debug("Vector store created and saved locally")
        return vector_store

    def query_llama_via_cli(self, input_text):
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
            return self.extract_relevant_answer(response)
        except subprocess.TimeoutExpired:
            process.kill()
            logging.error("Timeout for the model request")
            return "Timeout for the model request"
        except Exception as e:
            logging.error(f"An unexpected error has occurred: {str(e)}")
            return f"An unexpected error has occurred: {str(e)}"

    def extract_relevant_answer(self, full_response):
        """Extract the relevant response from the full model response."""
        logging.debug("Extracting relevant answer from model response")
        response_lines = full_response.splitlines()
        if response_lines:
            return "\n".join(response_lines).strip()
        return "No answer received"

    def get_conversational_chain(self, context, ques):
        """Create the input for the model based on the prompt and context."""
        logging.debug("Creating conversational chain")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an intelligent and helpful assistant. Your goal is to provide the most accurate and detailed answers 
                    possible to any question you receive. Use all available context to enhance your answers, and explain complex 
                    concepts in a simple manner. If additional information might help, suggest further areas for exploration. If the
                    answer is not available in the provided context, state this clearly and offer related insights when possible."""
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        input_text = f"Prompt: {prompt.format(input=ques)}\nContext: {context}\nQuestion: {ques}"
        response = self.query_llama_via_cli(input_text)
        return response

    def user_input(self, user_question):
        """Processes the user input and calls up the model."""
        logging.debug("Processing user input")
        context = self.pdf_text
        response = self.get_conversational_chain(context, user_question)
        return response

    def ask_question(self, user_question):
        """Ask a question based on the loaded PDF."""
        logging.debug(f"Asking question: {user_question}")
        response = self.user_input(user_question)
        return response


from logzero import logger

# Example of usage:
if __name__ == "__main__":
    pdf_file_path = "/home/hp/Downloads/ddpm.pdf"  # Set the correct path to your PDF file
    pdf_qa = PdfQuestionAnswer(pdf_file_path)
    
    

    # Now you can ask multiple questions after initializing the class:
    question_1 = "What is the main topic of the document?"
    question_2 = "Can you explain the second section in more detail?"
    # Ask questions and get responses
    logger.info(pdf_qa.ask_question(question_1))
    logger.info(pdf_qa.ask_question(question_2))
