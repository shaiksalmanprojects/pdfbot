import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

import streamlit as st
from streamlit_option_menu import option_menu

# Function to add custom CSS
def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Global styles */
        body {
            background-color: #f5f7fa;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background: linear-gradient(120deg, #f5f7fa, #dcecfb);
        }
        .header {
            text-align: center;
            color: #00274d;
        }
        .header h1 {
            font-size: 3rem;
            font-weight: bold;
        }
        .subheader {
            text-align: center;
            color: #4d4d4d;
            margin-bottom: 30px;
        }
        /* Sidebar styles */
        .sidebar .sidebar-content {
            background-color: #00274d;
            color: #ffffff;
        }
        /* Buttons */
        .button {
            display: inline-block;
            padding: 10px 20px;
            font-size: 1.2rem;
            color: #ffffff;
            background-color: #007bff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            margin-top: 10px;
        }
        .button:hover {
            background-color: #0056b3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Splits the text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up the conversational chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Processes user question and fetches a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("### Reply:", response["output_text"])

def main():
    # Apply custom styles
    apply_custom_styles()

    # App layout
    st.markdown("<div class='header'><h1>Explore the World of AI 📚</h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'><p>Upload your PDFs and ask questions seamlessly!</p></div>", unsafe_allow_html=True)

    # Sidebar menu
    with st.sidebar:
        st.markdown("### Menu:")
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Upload PDF", "Chat"],
            icons=["house", "upload", "chat"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#00274d"},
                "icon": {"color": "white", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "#dcecfb",
                },
                "nav-link-selected": {"background-color": "#0056b3"},
            },
        )

    # Page content
    if selected == "Home":
        st.markdown("### Welcome to the AI Chat with PDF!")
        st.write(
            "This tool lets you interact with your PDFs using AI. Start by uploading a file in the sidebar."
        )
    elif selected == "Upload PDF":
        st.markdown("### Upload Your PDF")
        pdf_files = st.file_uploader("Select PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("Process PDFs"):
            if pdf_files:
                st.success("PDFs uploaded and processed successfully!")
            else:
                st.error("Please upload at least one PDF file.")
    elif selected == "Chat":
        st.markdown("### Chat with Your PDF")
        user_question = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if user_question:
                user_input(user_question)
            else:
                st.error("Please type a question.")

if __name__ == "__main__":
    main()
