
import os
import json
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# os.environ["GROQ_API_KEY"] = "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"

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Store text chunks as vector embeddings using FAISS
def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to store generated questions in a JSON file
def store_questions(questions, chapter):
    with open(f'questions_chapter_{chapter}.json', 'w') as f:
        json.dump(questions, f)

# Retry decorator to handle API quota limits
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_questions(text_chunk: str, chapter: int) -> List[str]:
    from langchain_google_genai import ChatGoogleGenerativeAI  # Correct import
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    

    llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.5
        )

    
    # Enhanced prompt to generate varied, deep questions that probe user understanding
    prompt = PromptTemplate(
        input_variables=["text", "chapter"],
        template="""
        Based on the following chapter text, generate 5 diverse, thought-provoking questions 
        that assess the user's understanding of the key concepts and encourage real-world application.
        Tailor the questions to follow the natural progression of ideas in the chapter:

        Chapter {chapter}:
        {text}

        Questions:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    questions_text = chain.run(text=text_chunk, chapter=chapter)
    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
    return questions[:5]  # Limit to top 5 questions

# Function to generate personalized learning plan based on user responses
def generate_personalized_plan(vector_store: FAISS, chapter: int, responses: List[str]):
    plan = []
    for i, response in enumerate(responses):
        results = vector_store.similarity_search_with_score(
            query=response, k=100, filter={"chapter": chapter, "type": "response"}
        )
        question = results[0][0].metadata.get('question', '')
        similarity_score = 1 - results[0][1]  # Convert distance to similarity score
        
        # Dynamic feedback and actions based on response similarity
        if similarity_score < 0.6:
            plan.append(f"Chapter {chapter}: You seem to be struggling with '{question}'. "
                        f"Consider revisiting the chapter and applying the concept in a practical scenario. "
                        f"Suggested action: Re-read section XYZ and experiment with a related case study.")
        elif 0.6 <= similarity_score < 0.8:
            plan.append(f"Chapter {chapter}: You have a moderate grasp on '{question}'. "
                        f"Try discussing this concept with a peer or writing a reflective summary to solidify your understanding. "
                        f"Suggested action: Summarize this chapter in 3 sentences and explore its application in a current project.")
        else:
            plan.append(f"Chapter {chapter}: Excellent understanding of '{question}'. "
                        f"Challenge yourself by applying this knowledge in a new context or exploring an advanced application. "
                        f"Suggested action: Use this concept in a real-world project or research.")

    return plan

# Function to store user responses and analyze comprehension
def store_responses(vector_store: FAISS, chapter: int, question: str, response: str):
    response_embedding = embeddings.embed_query(response)
    vector_store.add_texts(
        texts=[response],
        metadatas=[{"chapter": chapter, "question": question, "type": "response"}],
        embeddings=[response_embedding]
    )
    
    # Analyze response and provide feedback based on the similarity score
    similarity_scores = vector_store.similarity_search_with_score(response, k=1)
    top_score = similarity_scores[0][1]  # Best match score
    
    if top_score < 0.6:
        st.write("Your response indicates that you may need more clarity on this topic. Consider revisiting the chapter.")
    elif 0.6 <= top_score < 0.8:
        st.write("You seem to have a fair understanding. To reinforce learning, try applying the concept in a new scenario.")
    else:
        st.write("Great! You have a strong grasp of the topic. Now you can explore more advanced concepts.")

# Streamlit App
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📚", layout="wide")
    
    st.title("Chat with Your PDF using AI 💁")
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []

    # Upload PDF Files
    st.subheader("Upload Your PDF Files")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Submit & Process") or st.session_state.processed:
        if not st.session_state.processed:
            with st.spinner("Processing..."):
                # Process the PDF
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)  # Create vector store
                
                st.success("PDF processed and embedded successfully!")
                
                # Automatically generate and store questions for each chapter
                for chapter, chunk in enumerate(text_chunks, start=1):
                    questions = generate_questions(chunk, chapter)
                    store_questions(questions, chapter)
                    st.write(f"Generated and stored questions for Chapter {chapter}")

                # Store the processed state and vector store in session state
                st.session_state.processed = True
                st.session_state.vector_store = vector_store
                st.session_state.text_chunks = text_chunks

        # Select chapter to explore
        chapter_selected = st.selectbox("Select a Chapter to Explore", range(1, len(st.session_state.text_chunks) + 1))
        st.subheader(f"Chapter {chapter_selected} Content Preview")
        st.write(st.session_state.text_chunks[chapter_selected - 1][:500] + "...")  # Show first 500 characters of the chapter

        # Load generated questions
        with open(f'questions_chapter_{chapter_selected}.json') as f:
            questions = json.load(f)

        st.subheader("Generated Questions")
        responses = []
        for i, question in enumerate(questions, 1):
            response = st.text_area(f"{i}. {question}")
            responses.append(response)

        if st.button("Submit Responses"):
            for i, response in enumerate(responses):
                store_responses(st.session_state.vector_store, chapter_selected, questions[i], response)

            st.success("Responses submitted!")

            # Generate personalized action plan based on responses
            personalized_plan = generate_personalized_plan(st.session_state.vector_store, chapter_selected, responses)
            st.subheader("Personalized Learning Action Plan")
            for i, action in enumerate(personalized_plan, 1):
                st.write(f"{i}. {action}")

if __name__ == "__main__":
    main()
