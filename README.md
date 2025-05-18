# 🩺 MIMIC-QA Chatbot

## 📌 프로젝트 개요

**MIMIC-QA Chatbot**은 MIMIC 데이터셋을 처음 다루거나 연구에 활용하려는 사용자들을 위해 설계된 챗봇입니다.  
MIMIC 데이터셋은 구조가 복잡하고 관련 문서가 흩어져 있기 때문에, 이를 쉽게 탐색할 수 있도록 LangChain 기반의 RAG(Retrieval-Augmented Generation) 구조를 사용하여 챗봇 형태로 구현했습니다.

이 프로젝트는 **로컬 GPU 환경 (3090 / 15GB VRAM)** 에서 실행 가능하며,  
**Ollama**의 로컬 LLM을 통해 LangChain과 연동되어 동작합니다.

---

## 📚 데이터 구성

챗봇은 다음과 같은 파일들을 기반으로 지식베이스를 구축하였습니다:

- `33개` PDF 문서  
- `2개` Word (.docx) 문서  
- `7개` 텍스트 (.txt) 파일  

이 문서들은 다음 출처에서 수집되었습니다:

- MIMIC 공식 문서  
- SQL 쿼리 예시  
- MIMIC 관련 활용 논문들  

이 모든 문서는 벡터화되어 로컬 캐시에 저장되며, 사용자의 질문에 대한 정보를 검색할 때 사용됩니다.

---

## ⚙️ 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/mimic-qa-chatbot.git
cd mimic-qa-chatbot
```

### 2. Conda 환경 구성

```bash
conda env create -f environment.yaml
conda activate langchain
```

### 3. Ollama 설치 (Ubuntu 기준)
로컬 LLM 실행을 위한 Ollama 설치 방법은 ollama_install.md에 포함되어 있습니다.
공식 사이트: https://ollama.com

## 🚀 실행 방법

### ✅ Streamlit 실행

- **로컬 Ollama 기반 실행:**

```bash
cd app
streamlit run app.py
```

- **Gemini API 기반 실행:**
```bash
cd app
streamlit run app_gemini.py
```

### ✅ Gradio  실행

- **기본 Gradio UI 실행:**

```bash
cd app
python gradio_app.py
```

- **히스토리(이전 질문 기억) 포함 Gradio UI 실행:**
```bash
cd app
python gradio_his_app.py
```

## 💡 주요 기능

- 🔍 **자연어 질문 응답**  
  MIMIC 관련 문서를 기반으로 자연어 질문에 대해 정확한 답변 제공

- 🧠 **문서 벡터화 및 로컬 캐시**  
  총 33개의 PDF, 2개의 Word, 7개의 텍스트 파일을 임베딩하여 dictionary 형태로 벡터 저장  
  최초 1회 임베딩 후 캐시에 저장되며, 빠른 응답을 지원

- 💬 **대화 히스토리 기능**  
  이전 질문과 답변을 기억하여 연속적인 대화 흐름 유지

- 🧱 **로컬 LLM 추론 (Ollama 사용)**  
  Langchain 기반 RAG + Ollama LLM으로 GPU 로컬 환경에서 실행 가능  
  인터넷 연결 없이도 고성능 LLM 작동

- ❌ **모르는 질문에 대한 회피 설정**  
  명확한 정보가 없는 경우 무작정 답변하지 않도록 프롬프트 설계

- ⚙️ **멀티 인터페이스 지원**  
  Streamlit 또는 Gradio 기반 UI 제공  
  사용 환경에 따라 원하는 방식으로 실행 가능

## 🖥️ 시스템 요구 사항

- **운영체제**: Ubuntu 20.04 이상 권장
- **GPU**: 최소 15GB VRAM 이상 (예: RTX 3090)
- **RAM**: 32GB 이상
- **저장공간**: 최소 5GB 여유 공간
- **Python 버전**: 3.9 이상
- **추가 요구사항**:
  - Conda 환경 사용 권장
  - Ollama 설치 필요 (로컬 LLM 구동 시)

---

## 📂 폴더 구조 예시

```bash
mimic-qa-chatbot/
├── docs/                      # 문서 자료 모음 (PDF, DOCX, TXT 등)
│   ├── *.pdf
│   ├── *.docx
│   └── *.txt
├── app/                          # Application scripts
│   ├── chroma_db/                # 문서 임베딩 캐시 저장소
│   ├── app.py                    # Streamlit 로컬 실행 파일
│   ├── app_gemini.py             # Gemini API 기반 실행 파일
│   ├── gradio_app.py             # Gradio 기본 실행 파일
│   └── gradio_his_app.py         # Gradio + 대화 히스토리 실행 파일
├── environment.yaml          # Conda 환경 구성 파일
└── README.md                 # 프로젝트 설명서
