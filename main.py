import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# --- 1. โหลด Environment Variables และตั้งค่าเริ่มต้น ---
load_dotenv()

# ตรวจสอบว่ามี API Key ในระบบหรือไม่
if not os.getenv("OPENROUTER_API_KEY"):
    raise EnvironmentError("กรุณาตั้งค่า OPENROUTER_API_KEY ในไฟล์ .env ด้วยครับ")

CHROMA_DB_DIR = "chroma_db_store"
MODEL_NAME = "intfloat/multilingual-e5-large" #-- intfloat/multilingual-e5-base // intfloat/multilingual-e5-small
LLM_MODEL_ON_OPENROUTER = "meta-llama/llama-3-8b-instruct" # หรือ "mistralai/mixtral-8x7b-instruct"

print("--- กำลังเริ่มต้น Backend Server ---")

# --- 2. เตรียมส่วนประกอบของ RAG (โหลดทุกอย่างรอไว้ในหน่วยความจำ) ---

# โหลด Embedding function (ใช้ CPU ก็เพียงพอสำหรับการ query)
print("กำลังโหลด Embedding model...")
embedding_function = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# โหลด Vector Store ที่เราสร้างไว้
print("กำลังโหลด Vector Store จาก ChromaDB...")
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_function
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # ตั้งค่าให้ดึงข้อมูลที่เกี่ยวข้องที่สุดมา 5 chunks

# สร้าง LLM client ที่ชี้ไปที่ OpenRouter
print(f"กำลังเชื่อมต่อกับ LLM: {LLM_MODEL_ON_OPENROUTER} ผ่าน OpenRouter...")
llm = ChatOpenAI(
    model=LLM_MODEL_ON_OPENROUTER,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2, # ลดความเพี้ยนของคำตอบ
    max_tokens=1024
)

# สร้าง Prompt Template
# นี่คือส่วนที่สำคัญที่สุดในการ "สอน" ให้ AI ตอบตามที่เราต้องการ
PROMPT_TEMPLATE = """
คุณคือผู้ช่วยช่างเทคนิค AI อัจฉริยะชื่อ "ช่างรู้" (Chang-Roo) หน้าที่ของคุณคือตอบคำถามเกี่ยวกับงานซ่อมบำรุง
โดยใช้ข้อมูลจาก "เอกสารอ้างอิง" ที่ให้มาเท่านั้น ห้ามตอบนอกเหนือจากข้อมูลนี้เด็ดขาด

พูดคุยด้วยภาษาไทยที่เป็นกันเอง สุภาพ และเข้าใจง่ายเหมือนคุยกับเพื่อนร่วมงาน
เริ่มต้นคำตอบด้วยการสรุปประเด็นสำคัญก่อน แล้วค่อยอธิบายรายละเอียดตามขั้นตอน

เอกสารอ้างอิง:
{context}

คำถามจากช่างเทคนิค:
{question}

คำตอบจากช่างรู้:
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
print("--- Backend Server พร้อมให้บริการ! ---")

# --- 3. สร้าง FastAPI App และ API Endpoint ---
app = FastAPI()

# Pydantic model สำหรับรับข้อมูล JSON ที่ส่งมา
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Endpoint สำหรับรับคำถามและตอบกลับโดยใช้ RAG
    """
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="ไม่พบคำถาม (Question is empty)")

        print(f"\nได้รับคำถาม: {question}")

        # 1. Retrieval: ค้นหาเอกสารที่เกี่ยวข้อง
        relevant_docs = retriever.invoke(question)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 2. Augment & 3. Generation: สร้าง Prompt และส่งให้ LLM
        chain = prompt | llm
        response = chain.invoke({
            "context": context_text,
            "question": question
        })
        
        answer = response.content
        print(f"คำตอบที่สร้างโดย AI: {answer}")

        return {"answer": answer}

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. Endpoint สำหรับการทดสอบว่า Server ทำงานหรือไม่ ---
@app.get("/")
def read_root():
    return {"message": "Maintenance AI Assistant Backend is running!"}

# .\.venv\Scripts\activate
# uvicorn main:app --reload