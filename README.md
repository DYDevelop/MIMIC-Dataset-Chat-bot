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
