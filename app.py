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
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
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

# --- 2. تحميل المكتبة وتهيئة ChromaDB (نسخة متينة وآمنة) ---
all_knowledge = []
KB_PATH = "library_knowledge"
chroma_collection = None
db_status = 'not_started' # 'not_started', 'initializing', 'ready', 'failed'
db_lock = threading.Lock() # قفل لمنع التهيئة المتزامنة

def initialize_knowledge_base():
    """
    تقوم ببناء قاعدة البيانات عند الحاجة فقط (Lazy Loading) بطريقة آمنة.
    """
    global all_knowledge, chroma_collection, db_status

    with db_lock:
        if db_status == 'ready':
            print("✅ قاعدة البيانات جاهزة بالفعل.")
            return
        if db_status == 'initializing':
            print("⏳ التهيئة جارية بالفعل، يرجى الانتظار...")
            # انتظر حتى تكتمل التهيئة الحالية
            while db_status == 'initializing':
                time.sleep(1)
            return

        # ابدأ التهيئة
        db_status = 'initializing'
        print("🚀 بدء بناء قاعدة البيانات الدلالية لأول مرة... هذا سيستغرق وقتاً.")

    try:
        start_time = time.time()

        # 1. تحميل ملفات JSON
        all_knowledge = []
        if os.path.exists(KB_PATH):
            for filename in sorted(os.listdir(KB_PATH)):
                if filename.endswith(".json"):
                    with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                        all_knowledge.extend(json.load(f))
        
        if not all_knowledge:
            print("⚠️ لم يتم العثور على ملفات معرفة. لن تتم تهيئة قاعدة البيانات.")
            with db_lock:
                db_status = 'failed' # الفشل لأن لا توجد بيانات
            return

        # 2. تهيئة ChromaDB (صريحاً في الذاكرة للخطة المجانية)
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
            "page_pdf": str(unit.get("page_pdf", "--")) # الاحتفاظ به كنص مؤقتاً
        } for unit in all_knowledge]
        ids = [unit.get("unit_id", f"id_{i}") for i, unit in enumerate(all_knowledge)]

        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            # إصلاح 1: التعامل الآمن مع استجابة API
            try:
                response = genai.embed_content(model=EMBEDDING_MODEL, content=batch_docs)
                embeddings = response.get("embedding", response.get("embeddings"))
                if not embeddings:
                    raise ValueError("Embeddings not found in API response.")
            except Exception as e:
                print(f"❌ فشل في الحصول على المتجهات: {e}")
                with db_lock:
                    db_status = 'failed'
                return
            
            # إصلاح 7: التحقق من تطابق أحجام الدفعات
            if len(embeddings) != len(batch_docs):
                print(f"❌ عدم تطابق في حجم الدفعة: {len(embeddings)} embeddings vs {len(batch_docs)} docs.")
                with db_lock:
                    db_status = 'failed'
                return

            chroma_collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=metadatas[i:i+batch_size]
            )
            print(f"✅ تمت معالجة {min(i + batch_size, len(documents))} من أصل {len(documents)} وحدة.")

        end_time = time.time()
        print(f"🎉 اكتمل بناء قاعدة البيانات في {end_time - start_time:.2f} ثانية.")
        with db_lock:
            db_status = 'ready'

    except Exception as e:
        print(f"❌ حدث خطأ عام أثناء تهيئة قاعدة البيانات: {e}")
        with db_lock:
            db_status = 'failed'

# --- 3. محرك البحث الدلالي ---
def semantic_search(query, collection, n_results=6):
    if not collection or db_status != 'ready':
        return []
    response = genai.embed_content(model=EMBEDDING_MODEL, content=query)
    query_embedding = response.get("embedding", response.get("embeddings"))
    if not query_embedding:
        print("❌ فشل في الحصول على متجه للاستعلام.")
        return []
        
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    
    final_results = []
    for i in range(len(results['ids'][0])):
        # إصلاح 2: التحويل الآمن لـ page_pdf
        page_pdf_str = results['metadatas'][0][i].get('page_pdf', '--')
        try:
            page_pdf_int = int(page_pdf_str)
        except (ValueError, TypeError):
            page_pdf_int = 0 # قيمة افتراضية

        final_results.append({
            "unit_id": results['ids'][0][i],
            "content": results['documents'][0][i],
            "author": results['metadatas'][0][i].get('author', '--'),
            "book": results['metadatas'][0][i].get('book', '--'),
            "part": results['metadatas'][0][i].get('part', '--'),
            "page_pdf": page_pdf_int
        })
    return final_results

# --- 4. نقطة الاتصال (مع التهيئة عند الطلب والتحقق من الحالة) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # إصلاح 3 و 6: التحقق من حالة قاعدة البيانات قبل البدء
        if db_status == 'failed':
            return jsonify({"answer": "❌ فشلت تهيئة قاعدة المعرفة. يرجى مراجعة سجلات الخادم."}), 503

        initialize_knowledge_base()

        if db_status != 'ready':
             return jsonify({"answer": "⏳ قاعدة المعرفة قيد التجهيز حالياً، يرجى المحاولة بعد قليل."}), 503

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
        app.logger.error(f"An error occurred in /ask: {e}", exc_info=True)
        return jsonify({"answer": f"❌ خطأ تقني غير متوقع: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
