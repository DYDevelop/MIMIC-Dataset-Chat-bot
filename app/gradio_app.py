import gradio as gr
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 문서 로딩
def load_documents(folder_path="../DY_Applied/*.pdf"):
    pdf_files = glob(folder_path)
    all_docs = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs = loader.load_and_split()
        all_docs.extend(docs)
    return all_docs

# VectorStore 생성
def build_vectorstore(docs):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vectorstore

# QA Chain 생성
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    model = ChatOllama(
        # model="deepseek-r1:14b"
        # model="mistral-small3.1:latest"
        model="gemma3:12b"
        # model="benedict/linkbricks-llama3.1-korean:8b"
        )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 

    <context>
    {context}
    </context>

    Answer the following question:

    {question}""")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )
    return chain

# 초기화
print("📄 Loading PDFs and creating vectorstore...")
docs = load_documents()
vectorstore = build_vectorstore(docs)
qa_chain = build_qa_chain(vectorstore)
print("✅ Ready to serve!")

# Gradio 인터페이스
def answer_question(question):
    return qa_chain.invoke(question)

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="💬 Ask a question based on the loaded PDFs"),
    outputs=gr.Textbox(label="📘 Answer"),
    title="MIMIC PDF QA Bot",
    description="Ask about papers that used MIMIC dataset (e.g., length of stay, discharge prediction, etc.)"
)

demo.launch(server_port=7860)
