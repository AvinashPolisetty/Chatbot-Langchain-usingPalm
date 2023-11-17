import streamlit as st
from main import create_vector_db,get_qa_chain

st.header("Q&A chatbot")

button=st.button("create knowledgebase")
if button:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain=get_qa_chain()
    response=chain(question)

    st.header("Answer")
    st.write(response['result'])
    