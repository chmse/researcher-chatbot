
import os
import json
import re
import time
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.api_core import exceptions
import chromadb

app = Flask(__name__)
CORS(app)

# --- 1. إعدادات جوجل Gemini ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
EMBEDDING_MODEL = 'models/embedding-001'

def get_model_name():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if '1.5-flash' in m: return m
        return models[0] if models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

MODEL_NAME = get_model_name()

# --- 2. تهيئة قاعدة البيانات (نفس خوارزميتك) ---
all_knowledge = []
KB_PATH = "library_knowledge"
chroma_collection = None
db_status = 'not_started'
db_lock = threading.Lock()

def initialize_knowledge_base():
    global all_knowledge, chroma_collection, db_status
    with db_lock:
        if db_status in ['ready', 'initializing']: return
        db_status = 'initializing'

    try:
        all_knowledge = []
        if os.path.exists(KB_PATH):
            for filename in sorted(os.listdir(KB_PATH)):
                if filename.endswith(".json"):
                    with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                        all_knowledge.extend(json.load(f))
        
        if not all_knowledge:
            db_status = 'failed'
            return

        chroma_client = chromadb.Client()
        try: chroma_client.delete_collection("knowledge_base")
        except: pass
        chroma_collection = chroma_client.create_collection("knowledge_base")

        documents = [u.get("content", "") for u in all_knowledge]
        metadatas = [{
            "author": u.get("author", "--"),
            "book": u.get("book", "--"),
            "part": u.get("part", "--"),
            "page_pdf": str(u.get("page_pdf", "--"))
        } for u in all_knowledge]
        ids = [f"id_{i}" for i in range(len(documents))]

        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            response = genai.embed_content(model=EMBEDDING_MODEL, content=batch_docs)
            embeddings = response.get("embedding", response.get("embeddings"))
            
            chroma_collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=metadatas[i:i+batch_size]
            )
        db_status = 'ready'
    except:
        db_status = 'failed'

def semantic_search(query, n_results=6):
    if db_status != 'ready': return []
    res = genai.embed_content(model=EMBEDDING_MODEL, content=query)
    embedding = res.get("embedding", res.get("embeddings"))
    results = chroma_collection.query(query_embeddings=[embedding], n_results=n_results)
    
    final = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        final.append({
            "content": results['documents'][0][i],
            "author": meta.get('author', '--'),
            "book": meta.get('book', '--'),
            "part": meta.get('part', '--'),
            "page_pdf": meta.get('page_pdf', '--')
        })
    return final

# --- 4. نقطة الاتصال (المحرر الأكاديمي الصارم) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        if db_status == 'failed': return jsonify({"answer": "❌ خطأ في قاعدة البيانات."}), 500
        initialize_knowledge_base()
        if db_status != 'ready': return jsonify({"answer": "⏳ جاري تهيئة المكتبة، يرجى المحاولة بعد قليل."}), 503

        data = request.json
        user_query = data.get("question")
        results = semantic_search(user_query)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة."})

        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n[مرجع: {i+1}] [ص: {u['page_pdf']}] [كتاب: {u['book']}] [مؤلف: {u['author']}] [ج: {u['part']}]\n{u['content']}\n"
        
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:
        1. المتن: انقل النص حرفياً بين علامتي تنصيص "" متبوعاً برقم المرجع [1].
        2. الحاشية (المراجع): رقم المرجع- اسم المؤلف، اسم الكتاب، الجزء، ص: الصفحة.
        3. الصرامة: ممنوع الإضافة من خارج النص. اطلب من الذكاء نقل كل النقاط (1، 2، 3...) إذا وجدت.
        
        النصوص:\n{ctx_text}\nسؤال المستخدم: {user_query}"""

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        for _ in range(3):
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests: time.sleep(15)
        
        return jsonify({"answer": "⚠️ الخادم مزدحم، حاول مجدداً."})
    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
