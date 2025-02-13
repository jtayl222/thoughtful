import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import (cosine_similarity)
import numpy as np

# Load and initialize data
def load_qa_data():
    qa_data = {
        "questions": [
            {
                "question": "What does the eligibility verification agent (EVA) do?",
                "answer": "EVA automates the process of verifying a patient's eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
            },
            {
                "question": "What does the claims processing agent (CAM) do?",
                "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
            },
            {
                "question": "How does the payment posting agent (PHIL) work?",
                "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
            },
            {
                "question": "Tell me about Thoughtful AI's Agents.",
                "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
            },
            {
                "question": "What are the benefits of using Thoughtful AI's agents?",
                "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
            }
        ]
    }
    return qa_data

class SupportAgent:
    def __init__(self, qa_data):
        self.qa_data = qa_data
        self.questions = [qa["question"] for qa in qa_data["questions"]]
        self.answers = [qa["answer"] for qa in qa_data["questions"]]
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
    
    def find_best_match(self, user_question, threshold=0.3):
        # Transform user question
        user_vector = self.vectorizer.transform([user_question])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, self.question_vectors)[0]
        
        # Find best match
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]
        
        if best_match_score >= threshold:
            return self.answers[best_match_index], best_match_score
        else:
            return (
                "I apologize, but I don't have specific information about that. "
                "Please try asking about EVA, CAM, PHIL, or general questions about Thoughtful AI's agents.",
                0.0
            )

def main():
    st.title("Thoughtful AI Support Agent")
    st.write("Ask me anything about Thoughtful AI's agents (EVA, CAM, PHIL)!")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize agent
    qa_data = load_qa_data()
    agent = SupportAgent(qa_data)
    
    # User input
    user_question = st.text_input("Your question:", key="user_input")
    
    if st.button("Ask"):
        if user_question:
            # Get response from agent
            answer, confidence = agent.find_best_match(user_question)
            
            # Add to chat history
            st.session_state.chat_history.append({"user": user_question, "bot": answer})
    
    # Display chat history
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Agent:** {chat['bot']}")
        st.write("---")

if __name__ == "__main__":
    main()
