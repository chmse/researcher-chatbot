import os
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.api_core import exceptions

### ### 1. إضافة مكتبة ChromaDB ###
import chromadb

app = Flask(__name__)
CORS(app)

# --- 1. إعدادات جوجل Gemini واكتشاف النموذج ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# تعريف نموذج التضمين (Embedding Model) كمتغير عام
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

# --- 2. تحميل المكتبة وتهيئة ChromaDB ---
all_knowledge = []
KB_PATH = "library_knowledge"
chroma_collection = None # سيتم تعيينه عند بدء التشغيل

def initialize_knowledge_base():
    """
    هذه الدالة تقوم بتحميل ملفات JSON وملء قاعدة بيانات ChromaDB.
    ستستغرق هذه العملية وقتاً طويلاً في كل مرة يتم فيها إعادة تشغيل التطبيق.
    """
    global all_knowledge, chroma_collection
    
    print("🚀 بدء تحميل المكتبة وتهيئة قاعدة البيانات الدلالية... قد يستغرق هذا بعض الوقت.")
    start_time = time.time()

    # 1. تحميل ملفات JSON كما في السابق
    all_knowledge = []
    if os.path.exists(KB_PATH):
        for filename in sorted(os.listdir(KB_PATH)):
            if filename.endswith(".json"):
                with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
    
    if not all_knowledge:
        print("⚠️ لم يتم العثور على ملفات معرفة. لن تتم تهيئة قاعدة البيانات.")
        return

    # 2. تهيئة عميل ChromaDB في الذاكرة (لا يستخدم قرصاً)
    chroma_client = chromadb.Client()
    # حذف المجموعة القديمة إذا وجدت لضمان بيانات جديدة
    try:
        chroma_client.delete_collection("knowledge_base")
    except:
        pass
    
    chroma_collection = chroma_client.create_collection("knowledge_base")

    # 3. تحويل كل وحدة معرفية إلى متجه وإضافتها إلى قاعدة البيانات
    documents = []
    metadatas = []
    ids = []

    for unit in all_knowledge:
        documents.append(unit.get("content", ""))
        metadatas.append({
            "author": unit.get("author", "--"),
            "book": unit.get("book", "--"),
            "part": unit.get("part", "--"),
            "page_pdf": str(unit.get("page_pdf", "--"))
        })
        ids.append(unit.get("unit_id", f"id_{len(ids)}"))

    # تقسيم المهام إلى دفعات لتجنب أخطاء الذاكرة أو حدود الطلبات
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        # الحصول على المتجهات (Embeddings) من جوجل
        response = genai.embed_content(model=EMBEDDING_MODEL, content=batch_docs)
        embeddings = response["embedding"]
        
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        chroma_collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_metadatas
        )
        print(f"✅ تمت معالجة {min(i + batch_size, len(documents))} من أصل {len(documents)} وحدة.")

    end_time = time.time()
    print(f"🎉 اكتملت تهيئة قاعدة البيانات الدلالية في {end_time - start_time:.2f} ثانية.")

# استدعاء دالة التهيئة عند بدء تشغيل التطبيق
initialize_knowledge_base()

# --- 3. محرك البحث الدلالي الجديد ---
def semantic_search(query, collection, n_results=6):
    """
    يستخدم قاعدة بيانات ChromaDB للبحث عن أقرب النصوص معنىً للاستعلام.
    """
    if not collection:
        return []
        
    # الحصول على المتجه الخاص بالاستعلام
    response = genai.embed_content(model=EMBEDDING_MODEL, content=query)
    query_embedding = response["embedding"]

    # البحث في قاعدة البيانات
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # إعادة تشكيل النتائج لتتوافق مع شكل الكود الأصلي
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

# --- 4. نقطة الاتصال (مع استخدام البحث الدلالي) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        ### ### استدعاء محرك البحث الجديد ###
        results = semantic_search(user_query, chroma_collection, n_results=6)
        
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n--- [معرف المرجع: {i+1}] ---\nالمؤلف: {u.get('author','--')} | الكتاب: {u.get('book','--')} | ج: {u.get('part','--')} | ص: {u.get('page_pdf','--')}\nالنص: {u['content']}\n"
        
        # الموجه (Prompt) المطور للاستفاضة والشمولية
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
