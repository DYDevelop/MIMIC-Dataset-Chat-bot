import gradio as gr
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
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
        model="gemma3:12b"  # 다른 모델로 변경 가능
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 채팅 히스토리를 포함한 프롬프트 템플릿
    rag_prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context and the conversation history to answer the question.
    If you don't know the answer, just say that you don't know.

    <context>
    {context}
    </context>

    <conversation_history>
    {history}
    </conversation_history>

    User: {question}
    Assistant:""")

    def run_chain(inputs):
        context = retriever.invoke(inputs["question"])
        formatted_context = format_docs(context)
        
        response = model.invoke(
            rag_prompt.format(
                context=formatted_context,
                history=inputs["history"],
                question=inputs["question"]
            )
        )
        
        return StrOutputParser().invoke(response)

    return run_chain

# 초기화
print("📄 Loading PDFs and creating vectorstore...")
docs = load_documents()
vectorstore = build_vectorstore(docs)
qa_chain = build_qa_chain(vectorstore)
print("✅ Ready to serve!")

# Gradio 인터페이스 (채팅 인터페이스로 변경)
def respond(message, chat_history):
    # 히스토리 텍스트 형식으로 변환
    history_text = ""
    for user_msg, bot_msg in chat_history:
        history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    
    # 질문과 히스토리를 체인에 전달
    bot_response = qa_chain({"question": message, "history": history_text})
    
    # 채팅 히스토리에 새 대화 추가 (Gradio Chatbot 형식에 맞게)
    chat_history.append((message, bot_response))
    
    # 새 메시지를 입력한 후 텍스트박스 비우기 위한 빈 문자열 반환
    return "", chat_history

# 채팅 인터페이스 설정
with gr.Blocks() as demo:
    gr.Markdown("# 📚 MIMIC PDF QA Bot with Memory")
    gr.Markdown("Ask about papers that used MIMIC dataset (e.g., length of stay, discharge prediction, etc.)")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="💬 질문을 입력하세요", placeholder="여기에 질문을 입력하세요...")
    clear = gr.Button("🧹 대화 내용 지우기")
    
    # 응답 함수의 출력을 msg와 chatbot 모두에 연결
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # 대화 내용 초기화
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_port=7860)