import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's questions."),
    ("user", "Question: {question}")
])

def generate_response(question, api_key, engine, temperature, max_tokens):
    llm = ChatGroq(
        model=engine,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

st.title("Groq Chatbot")
st.write("Powered by Groq")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    engine = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", 100, 4096, 1024, step=100)

question = st.text_area("Ask a Question", placeholder="Type your question here...")

if st.button("Generate Response", use_container_width=True):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = generate_response(question, api_key, engine, temperature, max_tokens)
                st.success("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")