import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- โค้ดกลับมาเรียบง่ายเหมือนตอนรันบนเครื่อง! ---
CHROMA_DB_DIR = "chroma_db_store"

# โหลด Environment Variables
load_dotenv()

# ที่เหลือเหมือนเดิมเกือบทั้งหมด...
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

print("--- กำลังเริ่มต้น Backend Server บน Render ---")

# เช็คว่า Vector DB อยู่ใน disk หรือยัง (สำหรับตอนรันครั้งแรก)
if not os.path.exists(CHROMA_DB_DIR):
    print(f"คำเตือน: ไม่พบ Vector DB ที่ '{CHROMA_DB_DIR}'.")
    print("โปรดรันสคริปต์ create_vector_db.py ผ่าน Render Shell ก่อนใช้งานครั้งแรก")
    # เราจะปล่อยให้ server รันต่อไป แต่ endpoint /ask จะใช้งานไม่ได้จนกว่าจะมี DB

print("กำลังโหลด Embedding model (อาจใช้เวลาสักครู่)...")
embedding_function = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print(f"กำลังโหลด Vector Store จาก: {CHROMA_DB_DIR}")
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_function
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("กำลังเชื่อมต่อกับ LLM ผ่าน OpenRouter...")
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
    max_tokens=1024
)

PROMPT_TEMPLATE = "..." # (คัดลอก Prompt Template เดิมของคุณมาใส่ตรงนี้)
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

print("--- Backend Server พร้อมให้บริการ! ---")

app = FastAPI()
# ... (ที่เหลือของ FastAPI endpoint /ask และ / เหมือนเดิมทุกประการ) ...
# (คัดลอกส่วน @app.post("/ask") และ @app.get("/") จากไฟล์เดิมมาวางต่อท้าย)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="ไม่พบคำถาม (Question is empty)")

        print(f"\nได้รับคำถาม: {question}")
        relevant_docs = retriever.invoke(question)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
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

@app.get("/")
def read_root():
    return {"message": "Maintenance AI Assistant Backend is running!"}

# .\.venv\Scripts\activate
# uvicorn main:app --reload