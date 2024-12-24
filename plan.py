import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Functions
def extract_text_from_pdfs(pdf_docs: List) -> str:
    """Extract text from uploaded PDF documents."""
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDFs: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 10000, overlap: int = 1000) -> List[str]:
    """Split text into manageable chunks."""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return []

def create_vector_store(text_chunks: List[str]) -> Optional[FAISS]:
    """Store text chunks as vector embeddings."""
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def save_questions_to_file(questions: List[str], chapter: int):
    """Save generated questions to a JSON file."""
    try:
        with open(f'questions_chapter_{chapter}.json', 'w') as file:
            json.dump(questions, file)
    except Exception as e:
        st.error(f"Error saving questions for Chapter {chapter}: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_questions(text_chunk: str, chapter: int) -> List[str]:
    """Generate questions for a given text chunk using an LLM."""
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
        prompt = PromptTemplate(
            input_variables=["text", "chapter"],
            template="""
            Based on the following chapter text, generate 5 diverse, thought-provoking questions 
            that assess the user's understanding of the key concepts and encourage real-world application:
            
            Chapter {chapter}:
            {text}

            Questions:
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        questions_text = chain.run(text=text_chunk, chapter=chapter)
        return [q.strip() for q in questions_text.split('\n') if q.strip()][:5]
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def generate_personalized_plan(vector_store: FAISS, chapter: int, responses: List[str]) -> List[str]:
    """Generate a personalized learning plan based on user responses."""
    plan = []
    for response in responses:
        try:
            results = vector_store.similarity_search_with_score(
                query=response, k=1, filter={"chapter": chapter, "type": "response"}
            )
            
            if not results:  # Check if no results were found
                plan.append(f"No relevant match found for '{response}'. Suggested action: Review the chapter content thoroughly.")
                continue

            # Extract the most relevant result
            question = results[0][0].metadata.get('question', 'Unknown question')
            similarity_score = 1 - results[0][1]

            # Create personalized suggestions based on similarity score
            if similarity_score < 0.6:
                plan.append(f"Struggling with '{question}'. Suggested action: Revisit the chapter.")
            elif 0.6 <= similarity_score < 0.8:
                plan.append(f"Moderate understanding of '{question}'. Suggested action: Summarize the key ideas.")
            else:
                plan.append(f"Excellent grasp of '{question}'. Suggested action: Apply this concept in a real-world scenario.")
        except Exception as e:
            st.error(f"Error generating personalized plan: {e}")
    return plan


# Streamlit App
def main():
    st.set_page_config(page_title="Interactive Learning Assistant", page_icon="📖", layout="wide")
    st.title("📚 Interactive Learning Assistant")

    # Upload PDFs
    st.sidebar.subheader("Upload Your PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    process_button = st.sidebar.button("Process PDFs")

    if process_button and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            raw_text = extract_text_from_pdfs(pdf_docs)
            if raw_text:
                text_chunks = split_text_into_chunks(raw_text)
                vector_store = create_vector_store(text_chunks)

                if vector_store:
                    st.success("PDFs processed successfully!")
                    st.session_state["vector_store"] = vector_store
                    st.session_state["text_chunks"] = text_chunks


    if "vector_store" in st.session_state and "text_chunks" in st.session_state:
        # Select chapter
        chapter_options = range(1, len(st.session_state["text_chunks"]) + 1)
        chapter = st.selectbox("Select Chapter", chapter_options)
        st.subheader(f"Chapter {chapter} Preview")
        st.write(st.session_state["text_chunks"][chapter - 1][:500] + "...")

        # Generate questions
        if st.button(f"Generate Questions for Chapter {chapter}"):
            questions = generate_questions(st.session_state["text_chunks"][chapter - 1], chapter)
            if questions:
                save_questions_to_file(questions, chapter)
                st.session_state[f"questions_{chapter}"] = questions

        if f"questions_{chapter}" in st.session_state:
            st.subheader("Generated Questions")
            responses = []
            for idx, question in enumerate(st.session_state[f"questions_{chapter}"], start=1):
                response = st.text_area(f"Q{idx}: {question}", key=f"response_{idx}")
                responses.append(response)

            if st.button("Submit Responses"):
                plan = generate_personalized_plan(
                    st.session_state["vector_store"], chapter, responses
                )
                st.subheader("Personalized Learning Plan")
                for idx, action in enumerate(plan, start=1):
                    st.write(f"{idx}. {action}")

if __name__ == "__main__":
    main()
