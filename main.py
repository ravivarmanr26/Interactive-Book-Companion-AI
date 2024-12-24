import os
import json
from typing import List, Optional, Dict
from dotenv import load_dotenv

import streamlit as st
import PyPDF2
from tenacity import retry, stop_after_attempt, wait_fixed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

class LearningAssistant:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Learning Assistant with embeddings
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        load_dotenv()
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

    def extract_text_from_pdfs(self, pdf_docs: List[st.uploaded_file_manager.UploadedFile]) -> str:
        """
        Advanced PDF text extraction with improved error handling
        
        Args:
            pdf_docs: List of uploaded PDF files
        
        Returns:
            Extracted text from PDFs
        """
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PyPDF2.PdfReader(pdf)
                text += " ".join(page.extract_text() or "" for page in pdf_reader.pages)
            except Exception as e:
                st.error(f"Error processing {pdf.name}: {e}")
        return text

    def split_text_into_chunks(
        self, 
        text: str, 
        chunk_size: int = 10000, 
        chunk_overlap: int = 1000
    ) -> List[str]:
        """
        Advanced text chunking with customizable parameters
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            return splitter.split_text(text)
        except Exception as e:
            st.error(f"Text chunking error: {e}")
            return []

    def create_vector_store(self, text_chunks: List[str]) -> Optional[FAISS]:
        """
        Create vector store with enhanced error handling
        
        Args:
            text_chunks: Text chunks to vectorize
        
        Returns:
            FAISS vector store or None
        """
        try:
            vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
            vector_store.save_local("faiss_index")
            return vector_store
        except Exception as e:
            st.error(f"Vector store creation error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def generate_questions(
        self, 
        text_chunk: str, 
        chapter: int, 
        num_questions: int = 5
    ) -> List[str]:
        """
        Generate diverse, thought-provoking questions
        
        Args:
            text_chunk: Text to generate questions from
            chapter: Chapter number
            num_questions: Number of questions to generate
        
        Returns:
            List of generated questions
        """
        prompt = PromptTemplate(
            input_variables=["text", "chapter"],
            template="""
            Generate {num_questions} diverse, thought-provoking questions 
            that assess understanding and encourage critical thinking:
            
            Chapter {chapter} Content:
            {text}

            Questions should:
            - Test comprehension
            - Encourage application of concepts
            - Promote critical analysis
            """
        )
        
        try:
            chain = LLMChain(llm=self.model, prompt=prompt)
            questions_text = chain.run(
                text=text_chunk, 
                chapter=chapter, 
                num_questions=num_questions
            )
            return [q.strip() for q in questions_text.split('\n') if q.strip()][:num_questions]
        except Exception as e:
            st.error(f"Question generation error: {e}")
            return []

    def generate_personalized_plan(
        self, 
        vector_store: FAISS, 
        chapter: int, 
        responses: List[str]
    ) -> List[str]:
        """
        Generate personalized learning plan
        
        Args:
            vector_store: FAISS vector store
            chapter: Chapter number
            responses: User's responses
        
        Returns:
            Personalized learning recommendations
        """
        plan = []
        for response in responses:
            try:
                results = vector_store.similarity_search_with_score(
                    query=response, 
                    k=1, 
                    filter={"chapter": chapter, "type": "response"}
                )
                similarity_score = 1 - results[0][1]
                
                recommendations = {
                    (0.0, 0.6): "Deep review needed. Consider breaking down complex concepts.",
                    (0.6, 0.8): "Good start. Practice applying concepts in different contexts.",
                    (0.8, 1.1): "Excellent understanding. Explore advanced applications and interdisciplinary connections."
                }
                
                plan.append(next(
                    rec for (low, high), rec in recommendations.items() 
                    if low <= similarity_score < high
                ))
            except Exception as e:
                st.error(f"Personalized plan generation error: {e}")
        
        return plan

def main():
    st.set_page_config(page_title="Intelligent Learning Assistant", page_icon="📖")
    assistant = LearningAssistant()

    st.title("📚 Intelligent Learning Assistant")
    
    # PDF Upload and Processing
    pdf_docs = st.sidebar.file_uploader(
        "Upload PDFs", 
        accept_multiple_files=True, 
        type=["pdf"]
    )
    
    if st.sidebar.button("Process PDFs") and pdf_docs:
        with st.spinner("Processing PDFs..."):
            raw_text = assistant.extract_text_from_pdfs(pdf_docs)
            text_chunks = assistant.split_text_into_chunks(raw_text)
            vector_store = assistant.create_vector_store(text_chunks)
            
            if vector_store:
                st.session_state["vector_store"] = vector_store
                st.session_state["text_chunks"] = text_chunks
                st.success("PDFs processed successfully!")

    # Learning Interaction
    if "vector_store" in st.session_state:
        chapter = st.selectbox(
            "Select Chapter", 
            range(1, len(st.session_state["text_chunks"]) + 1)
        )
        
        if st.button(f"Generate Questions for Chapter {chapter}"):
            questions = assistant.generate_questions(
                st.session_state["text_chunks"][chapter - 1], 
                chapter
            )
            st.session_state[f"chapter_{chapter}_questions"] = questions

        if f"chapter_{chapter}_questions" in st.session_state:
            responses = []
            for idx, question in enumerate(st.session_state[f"chapter_{chapter}_questions"], 1):
                response = st.text_area(f"Q{idx}: {question}")
                responses.append(response)

            if st.button("Submit Responses"):
                plan = assistant.generate_personalized_plan(
                    st.session_state["vector_store"], 
                    chapter, 
                    responses
                )
                st.subheader("Personalized Learning Plan")
                for idx, recommendation in enumerate(plan, 1):
                    st.write(f"{idx}. {recommendation}")

if __name__ == "__main__":
    main()