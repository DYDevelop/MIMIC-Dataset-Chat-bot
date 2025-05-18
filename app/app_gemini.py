import streamlit as st
from glob import glob
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mimic-chatbot-75c4155f036a.json"

# 앱 제목 설정
st.set_page_config(page_title="MIMIC QA Chat Bot", page_icon="📚")
st.title("📚 MIMIC QA Chat Bot")
st.markdown("Ask anything about MIMIC dataset (e.g., length of stay, discharge prediction, etc.)")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 현재 디렉토리에 db 폴더 생성
DB_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

# 초기화 함수
@st.cache_resource
def initialize_qa_system():
    # 문서 로딩
    folder_path = "../docs/*/*.pdf"
    pdf_files = glob(folder_path)
    
    if not pdf_files:
        st.error(f"PDF 파일을 찾을 수 없습니다: {folder_path}")
        st.stop()
    
    st.info(f"{len(pdf_files)}개의 PDF 파일을 로딩합니다...")
    
    all_docs = []
    for file in pdf_files:
        try:
            loader = PyPDFLoader(file)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"파일 로딩 중 오류: {file} - {e}")

    folder_path = "../docs/*/*.docx"
    word_files = glob(folder_path)
    
    if not word_files:
        st.error(f"WORD 파일을 찾을 수 없습니다: {folder_path}")
        st.stop()
    
    st.info(f"{len(word_files)}개의 WORD 파일을 로딩합니다...")
    
    for file in word_files:
        try:
            loader = Docx2txtLoader(file)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"파일 로딩 중 오류: {file} - {e}")

    folder_path = "../docs/*/*.txt"
    word_files = glob(folder_path)
    
    if not word_files:
        st.error(f"TXT 파일을 찾을 수 없습니다: {folder_path}")
        st.stop()
    
    st.info(f"{len(word_files)}개의 TXT 파일을 로딩합니다...")
    
    for file in word_files:
        try:
            loader = TextLoader(file)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"파일 로딩 중 오류: {file} - {e}")
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(all_docs)
    
    # 임베딩 및 벡터스토어 생성
    try:
        st.info("임베딩 모델을 초기화하고 벡터스토어를 생성합니다...")

        # 모델 설정
        GOOGLE_API_KEY = '****'
        genai.configure(api_key=GOOGLE_API_KEY)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Chroma 데이터베이스 설정 - 명시적으로 persist_directory 지정
        vector_store = Chroma.from_documents(split_docs, embeddings, persist_directory=DB_DIR)
        
        # 리트리버 설정
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        
        # 프롬프트 템플릿
        rag_prompt = ChatPromptTemplate.from_template("""
        You are a knowledgeable assistant specializing in clinical data analysis using the MIMIC (Medical Information Mart for Intensive Care) dataset.
        MIMIC is a large, freely-available database comprising de-identified health-related data associated with patients who stayed in critical care units of the Beth Israel Deaconess Medical Center. It includes structured data (e.g., demographics, lab results, medications, procedures, vitals) as well as unstructured data (e.g., clinical notes, discharge summaries).

        Your task is to answer questions based on the retrieved academic context and the conversation history, with a focus on topics involving the MIMIC dataset, such as patient outcomes, length of stay, mortality prediction, readmission, discharge timing, and machine learning applications.

        Use the following pieces of retrieved context and the conversation history to answer the question. 
        If the answer cannot be found in the context or previous history, respond that you don't know rather than making up an answer.

        <context>
        {context}
        </context>

        <conversation_history>
        {history}
        </conversation_history>

        User: {question}
        Assistant:""")
        
        # 답변 생성 함수
        def answer_question(question, history):
            # 컨텍스트 검색 및 포맷팅
            context = retriever.invoke(question)
            formatted_context = "\n\n".join(doc.page_content for doc in context)
            
            # 응답 생성
            response = model.generate_content(
                rag_prompt.format(
                    context=formatted_context,
                    history=history,
                    question=question
                )
            )
            
            return response.text
        
        return answer_question
        
    except Exception as e:
        import traceback
        st.error(f"초기화 중 오류가 발생했습니다: {e}")
        st.code(traceback.format_exc())
        return None

# 초기화 상태 표시
with st.spinner('📄 문서들을 로딩하고 벡터스토어를 생성하는 중...'):
    qa_function = initialize_qa_system()
    
    if qa_function:
        st.success("✅ 시스템이 준비되었습니다!")
    else:
        st.error("❌ 시스템 초기화에 실패했습니다.")
        st.stop()

# 저장된 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("💬 질문을 입력하세요"):
    # 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 대화 기록 텍스트 형식으로 변환
    history_text = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history_text += f"User: {msg['content']}\n"
        else:
            history_text += f"Assistant: {msg['content']}\n"
    
    # 봇 응답 생성 및 표시
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = qa_function(prompt, history_text)
            st.markdown(response)
    
    # 봇 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": response})

# 사이드바에 대화 초기화 버튼 추가
with st.sidebar:
    if st.button("🧹 대화 내용 지우기"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 모델 정보")
    st.markdown("- 사용 모델: `gemini-2.0-flash`")
    st.markdown("- 임베딩: `gemini/embedding-001`")

# streamlit run app_gemini.py
