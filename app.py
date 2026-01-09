import os
import json
import re
import time
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

# --- 2. تحميل المكتبة وتهيئة ChromaDB (الطريقة الجديدة) ---
all_knowledge = []
KB_PATH = "library_knowledge"
chroma_collection = None
is_db_initialized = False ### متغير جديد لتتبع الحالة

def initialize_knowledge_base():
    """
    تقوم ببناء قاعدة البيانات عند الحاجة فقط (Lazy Loading).
    """
    global all_knowledge, chroma_collection, is_db_initialized

    # إذا كانت قاعدة البيانات جاهزة، لا تفعل شيئاً
    if is_db_initialized:
        print("✅ قاعدة البيانات جاهزة بالفعل.")
        return

    print("🚀 بدء بناء قاعدة البيانات الدلالية لأول مرة... هذا سيستغرق وقتاً.")
    start_time = time.time()

    # 1. تحميل ملفات JSON
    all_knowledge = []
    if os.path.exists(KB_PATH):
        for filename in sorted(os.listdir(KB_PATH)):
            if filename.endswith(".json"):
                with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
    
    if not all_knowledge:
        print("⚠️ لم يتم العثور على ملفات معرفة.")
        is_db_initialized = True # منع إعادة المحاولة
        return

    # 2. تهيئة ChromaDB
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("knowledge_base")
    except:
        pass
    chroma_collection = chroma_client.create_collection("knowledge_base")

    # 3. إضافة البيانات
    documents = [unit.get("content", "") for unit in all_knowledge]
    metadatas = [{
        "author": unit.get("author", "--"),
        "book": unit.get("book", "--"),
        "part": unit.get("part", "--"),
        "page_pdf": str(unit.get("page_pdf", "--"))
    } for unit in all_knowledge]
    ids = [unit.get("unit_id", f"id_{i}") for i, unit in enumerate(all_knowledge)]

    batch_size = 50 # تقليل حجم الدفعة لتجنب الأخطاء
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        response = genai.embed_content(model=EMBEDDING_MODEL, content=batch_docs)
        embeddings = response["embedding"]
        
        chroma_collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"✅ تمت معالجة {min(i + batch_size, len(documents))} من أصل {len(documents)} وحدة.")

    end_time = time.time()
    print(f"🎉 اكتمل بناء قاعدة البيانات في {end_time - start_time:.2f} ثانية.")
    is_db_initialized = True

### ### التعديل الأهم: لا تستدعي الدالة عند بدء التشغيل!
# initialize_knowledge_base() 

# --- 3. محرك البحث الدلالي ---
def semantic_search(query, collection, n_results=6):
    if not collection:
        return []
    response = genai.embed_content(model=EMBEDDING_MODEL, content=query)
    query_embedding = response["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    
    final_results = []
    for i in range(len(results['ids'][0])):
        final_results.append({
            "unit_id": results['ids'][0][i],
            "content": results['documents'][0][i],
            "author": results['metadatas'][0][i]['author'],
            "book": results['metadatas'][0][i]['book'],
            "part": results['metadatas'][0][i]['part'],
            "page_pdf": int(results['metadatas'][0][i]['page_pdf'])
        })
    return final_results

# --- 4. نقطة الاتصال (مع التهيئة عند الطلب) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        ### ### التعديل الثاني: استدعاء التهيئة هنا ###
        # قبل البحث، تأكد من أن قاعدة البيانات جاهزة
        initialize_knowledge_base()

        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        results = semantic_search(user_query, chroma_collection, n_results=6)
        
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n--- [معرف المرجع: {i+1}] ---\nالمؤلف: {u.get('author','--')} | الكتاب: {u.get('book','--')} | ج: {u.get('part','--')} | ص: {u.get('page_pdf','--')}\nالنص: {u['content']}\n"
        
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:
        مهمتك صياغة إجابة 'شاملة'، 'موسعة'، و 'مرتبة' وفق الشروط الصارمة التالية:
        1. العبارة الاستهلالية: ابدأ الإجابة حصراً بـ: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        2. الاستقصاء: ابحث عن كل النقاط والتفاصيل (1، 2، 3، 4...) الواردة في النصوص المرفقة ولا تكتفِ بالملخص. انقل كل مفهوم مع شرحه الحرفي كما ورد.
        3. النقل الحرفي: انقل الجمل حرفياً كما وردت في المرجع، وضع كل نص منقول بين علامتي تنصيص مزدوجة "" متبوعاً برقم مرجع متسلسل [1]، ثم [2]، وهكذا.
        4. الترقيم المتسلسل: يجب أن يكون ترقيم المراجع في المتن متسلسلاً تصاعدياً (1، 2، 3...) حسب ظهورها في إجابتك.
        5. هيكل الفقرات: ابدأ كل نقطة أو فكرة جديدة في سطر جديد تماماً. استخدم العناوين الفرعية إذا كانت موجودة في النص.
        6. الحاشية: في نهاية الإجابة، اكتب عنواناً بارزاً (المراجع:) ثم سرد المراجع بالصيغة: رقم المرجع- اسم المؤلف، اسم الكتاب، الجزء، ص: رقم الصفحة.
        7. الصرامة: ممنوع تماماً إضافة أي معلومة خارجية.
        المادة العلمية المتاحة:
        {ctx_text}
        سؤال الباحث:
        {user_query}
        """

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        for _ in range(3):
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(15)
        
        return jsonify({"answer": "⚠️ الخادم مزدحم، يرجى المحاولة مرة أخرى."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
