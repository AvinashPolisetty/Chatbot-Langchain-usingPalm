from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()
import os
import openai



llm=GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0.0)
openai.api_key=os.environ['OPENAI_API_KEY']

instructor_embeddings = HuggingFaceBgeEmbeddings()
vectordb_filepath="FAISSDB_index"
def create_vector_db():

    loader=CSVLoader(file_path='common_faqs.csv',source_column='prompt',encoding='ISO-8859-1')
    data=loader.load()
    vector_db=FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    vector_db.save_local(vectordb_filepath)


def get_qa_chain():

    vectordb=FAISS.load_local(vectordb_filepath,embeddings=instructor_embeddings)

    retriever=vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})
    
    return chain








if __name__=="__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))

