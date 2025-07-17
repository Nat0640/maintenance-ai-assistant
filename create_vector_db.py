import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. กำหนดค่าต่างๆ ---
KNOWLEDGE_BASE_DIR = "knowledge_base" # ชื่อโฟลเดอร์ที่เก็บเอกสารของเรา
CHROMA_DB_DIR = "chroma_db_store" # ชื่อโฟลเดอร์ที่จะใช้เก็บ Vector Database

# เลือก Embedding Model ที่รองรับภาษาไทยได้ดี
# 'intfloat/multilingual-e5-large' เป็นโมเดลที่นิยมและมีประสิทธิภาพสูง
MODEL_NAME = "intfloat/multilingual-e5-large"
# หากต้องการโมเดลที่เล็กและเร็วขึ้น สามารถลองใช้ 'intfloat/multilingual-e5-base'

print("--- เริ่มกระบวนการสร้าง Vector Database ---")

# --- 2. โหลดเอกสารทั้งหมดจากโฟลเดอร์ ---
try:
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.*", show_progress=True)
    documents = loader.load()
    if not documents:
        print(f"ไม่พบเอกสารในโฟลเดอร์ '{KNOWLEDGE_BASE_DIR}'")
        exit()
    print(f"โหลดเอกสารทั้งหมด {len(documents)} ไฟล์เรียบร้อยแล้ว")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดเอกสาร: {e}")
    exit()

# --- 3. แบ่งเอกสารเป็นส่วนย่อยๆ (Chunking) ---
# เราแบ่งเพื่อให้แต่ละส่วนไม่ยาวเกินไปและมีความหมายในตัวเอง
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # ขนาดของแต่ละ chunk (ตัวอักษร)
    chunk_overlap=200 # ให้มีข้อความคาบเกี่ยวกันเล็กน้อยระหว่าง chunk
)
chunks = text_splitter.split_documents(documents)
print(f"แบ่งเอกสารออกเป็น {len(chunks)} chunks เรียบร้อยแล้ว")

# --- 4. สร้าง Embeddings และจัดเก็บลงใน ChromaDB ---
print("กำลังสร้าง Embeddings และจัดเก็บลงใน ChromaDB...")
print(f"ใช้โมเดล: {MODEL_NAME}")

# สร้าง embedding function จาก Hugging Face
# device='cpu' หมายถึงให้ใช้ CPU ในการคำนวณ (ถ้ามี GPU Nvidia สามารถเปลี่ยนเป็น 'cuda')
embedding_function = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# สร้างและบันทึก Vector Store ลงในดิสก์
# Chroma จะจัดการสร้างโฟลเดอร์และไฟล์ต่างๆ ให้เอง
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory=CHROMA_DB_DIR
)

print("\n--- กระบวนการเสร็จสมบูรณ์! ---")
print(f"Vector Database ถูกสร้างและบันทึกไว้ที่โฟลเดอร์ '{CHROMA_DB_DIR}' เรียบร้อยแล้ว")