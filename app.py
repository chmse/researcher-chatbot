import os
import json
import re
import time
import threading
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.api_core import exceptions
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات جوجل Gemini واكتشاف النموذج ---
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

# --- 2. تحميل المكتبة وتهيئة الفهرس الخفيف (Scikit-Learn) ---
all_knowledge = []
KB_PATH = "library_knowledge"
vector_index = None
all_embeddings = []
db_status = 'not_started' # 'ready', 'initializing', 'failed'
db_lock = threading.Lock()

def initialize_semantic_index():
    global all_knowledge, vector_index, all_embeddings, db_status
    with db_lock:
        if db_status in ['ready', 'initializing']: return
        db_status = 'initializing'
    
    try:
        # تحميل ملفات JSON
        all_knowledge = []
        if os.path.exists(KB_PATH):
            for filename in sorted(os.listdir(KB_PATH)):
                if filename.endswith(".json"):
                    with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                        all_knowledge.extend(json.load(f))
        
        if not all_knowledge:
            db_status = 'failed'
            return

        # توليد المتجهات (Embeddings) بدفعات
        documents = [u.get("content", "") for u in all_knowledge]
        batch_size = 50
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            res = genai.embed_content(model=EMBEDDING_MODEL, content=batch)
            embeddings = res.get("embedding", res.get("embeddings"))
            all_embeddings.extend(embeddings)
            
        # بناء فهرس البحث الخفيف
        all_embeddings = np.array(all_embeddings)
        vector_index = NearestNeighbors(n_neighbors=5, metric='cosine')
        vector_index.fit(all_embeddings)
        
        db_status = 'ready'
    except Exception as e:
        print(f"Error: {e}")
        db_status = 'failed'

# --- 3. محرك البحث الذكي (دلالي + خوارزمية التتبع التي طلبتها) ---
def normalize(text):
    if not text: return ""
    return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", text))).strip()

def advanced_semantic_library_search(query, top_k=3):
    """تدمج البحث عن طريق المعنى مع خوارزمية التتبع للقوائم"""
    if db_status != 'ready': return []
    
    # 1. البحث الدلالي عن أفضل "رؤوس أقلام"
    res = genai.embed_content(model=EMBEDDING_MODEL, content=query)
    query_vec = np.array(res['embedding']).reshape(1, -1)
    distances, indices = vector_index.kneighbors(query_vec, n_neighbors=top_k)
    
    final_indices = set()
    # 2. تطبيق خوارزمية التتبع (Look-ahead) لضمان شمولية القوائم والتعليل
    for idx in indices[0]:
        # جلب المقطع مع مسح تتابعي لضمان الشمولية (15 وحدة تالية) كما في كودك الأصلي
        for i in range(max(0, idx-1), min(len(all_knowledge), idx+15)):
            u_content = all_knowledge[i].get("content", "")
            # ضم المقطع إذا كان هو المختار أو يمثل تكملة لترقيم (قائمة)
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()):
                final_indices.add(i)
            # التوقف إذا ابتعدنا عن السياق ولم نجد ترقيماً
            if i > idx + 5 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()):
                break
                
    return [all_knowledge[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال (الموجه الأكاديمي الصارم - كما هو دون تغيير) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        initialize_semantic_index()
        if db_status != 'ready':
            return jsonify({"answer": "⏳ جاري تهيئة المكتبة دلالياً، يرجى الانتظار ثوانٍ ثم المحاولة."}), 503

        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        # البحث المطور
        results = advanced_semantic_library_search(user_query)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        # بناء سياق النصوص المرجعية
        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n--- [معرف المرجع: {i+1}] ---\nالمؤلف: {u.get('author','--')} | الكتاب: {u.get('book','--')} | ج: {u.get('part','--')} | ص: {u.get('page_pdf','--')}\nالنص: {u['content']}\n"
        
        # الموجه (Prompt) - تم الحفاظ عليه حرفياً كما طلبته
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:

        مهمتك صياغة إجابة 'مدمجة' و 'مرتبة' وفق الشروط الصارمة التالية:
        1. العبارة الاستهلالية: ابدأ الإجابة حصراً بـ: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        2. النقل الحرفي: انقل الجمل حرفياً كما وردت في المرجع، وضع كل نص منقول بين علامتي تنصيص مزدوجة "" متبوعاً برقم مرجع متسلسل [1]، ثم [2]، وهكذا.
        3. الترقيم المتسلسل: يجب أن يكون ترقيم المراجع في المتن متسلسلاً تصاعدياً (1، 2، 3...) حسب ظهورها في إجابتك، ولا تكرر نفس الرقم حتى لو كان من نفس الصفحة؛ بل أعطِ كل اقتباس رقماً جديداً.
        4. هيكل الفقرات: ابدأ كل نقطة أو عنصر أساسي في سطر جديد.
        5. الحاشية (المراجع): في نهاية الإجابة تماماً، اكتب عنواناً بارزاً (المراجع:) ثم اذكر المراجع المقابلة للأرقام المستخدمة بالصيغة: رقم المرجع- اسم المؤلف، اسم الكتاب، الجزء، ص: رقم الصفحة.
        6. الصرامة العلمية: ممنوع منعاً باتاً إضافة أي شرح أو تأويل من عندك. التوسع يكون فقط من خلال النصوص المرفقة.

        المادة العلمية المسترجعة من المكتبة:
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
        
        return jsonify({"answer": "⚠️ الخادم مزدحم حالياً، يرجى المحاولة بعد قليل."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
