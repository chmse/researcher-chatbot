import os
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.api_core import exceptions

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات جوجل Gemini واكتشاف النموذج ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_model():
    """البحث عن اسم النموذج الصحيح المتاح لتجنب خطأ 404"""
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if '1.5-flash' in m: return m
        return models[0] if models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

MODEL_NAME = get_model()

# --- 2. تحميل المكتبة الكاملة ---
all_knowledge = []
KB_PATH = "library_knowledge"

def load_library():
    global all_knowledge
    all_knowledge = []
    if os.path.exists(KB_PATH):
        for filename in os.listdir(KB_PATH):
            if filename.endswith(".json"):
                with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
    return len(all_knowledge)

load_library()

# --- 3. محرك البحث الذكي المطور (تتبع القوائم والتعليل) ---
def normalize(text):
    if not text: return ""
    return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", text))).strip()

def advanced_library_search(query, units, top_k=3):
    """خوارزمية البحث الموسعة لجلب القوائم والتعليل"""
    query_norm = normalize(query)
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        content_norm = normalize(unit.get("content", ""))
        score = sum(2 for kw in keywords if kw in content_norm)
        
        # ميزة إضافية: إذا كان المقطع يبدأ بترقيم مثل (1- أو 1) )
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', unit.get("content", "").strip()):
            score += 1
            
        if score > 0:
            # تحيز بسيط للصفحات الأولى
            score += (1 - (unit.get("page_pdf", 0) / 500))
            scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    # جلب أفضل النتائج مع مسح تتبعي (Look-ahead) لـ 12 وحدة تالية لضمان شمولية القوائم
    for _, idx in scored_indices[:top_k]:
        for i in range(max(0, idx-1), min(len(units), idx+12)):
            unit_content = units[i].get("content", "")
            # ضم المقطع إذا كان هو المختار أو يمثل تكملة لترقيم (قائمة)
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', unit_content.strip()):
                final_indices.add(i)
            # التوقف إذا ابتعدنا عن السياق ولم نجد ترقيماً أو كلمات بحث
            if i > idx + 4 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', unit_content.strip()):
                if not any(kw in normalize(unit_content) for kw in keywords):
                    break

    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال (API Endpoint) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"error": "No question"}), 400

        # أ. البحث الموسع
        results = advanced_library_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        # ب. بناء نص السياق الموثق
        ctx_text = ""
        for i, u in enumerate(results):
            page = u.get('page_pdf', '--')
            book = u.get('book', 'كتاب غير محدد')
            author = u.get('author', 'غير معروف')
            part = u.get('part', '--')
            ctx_text += f"\n[المرجع: {i+1}] [ص: {page}] [الكتاب: {book}] [المؤلف: {author}] [ج: {part}]\n{u['content']}\n"
        
        # ج. الموجه الصارم (كما طلبت دون اختصار)
        system_instruction = """
        أنت باحث أكاديمي متخصص في فكر الدكتور الحاج صالح.
        مهمتك: استخراج جميع النقاط والمفاهيم المتعلقة بسؤال المستخدم من النصوص المرفقة.
        
        القواعد الصارمة:
        1. الشمولية: استخرج كل المفاهيم (1، 2، 3، 4، 5...) المذكورة في النص ولا تكتفِ بالأولى فقط.
        2. النقل الحرفي: انقل الجمل كما هي داخل علامتي تنصيص ' '.
        3. المتن: انقل النص حرفياً بين علامتي تنصيص ' ' متبوعاً برقم المرجع [1].
        الحاشية: في النهاية، اذكر المراجع بالصيغة: رقم- المؤلف، الكتاب، الجزء، ص: الصفحة
        4. الروابط: استخدم روابط لغوية بسيطة للجمع بين النقاط الموزعة على صفحات مختلفة.
        5. ممنوع تماماً الإجابة من خارج المرفقات.
        """

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        # د. محاولة التوليد مع معالجة ضغط الطلبات
        for attempt in range(3):
            try:
                prompt = f"{system_instruction}\n\nنصوص المرجع:\n{ctx_text}\n\nسؤال المستخدم: {user_query}"
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(15)
        
        return jsonify({"answer": "⚠️ الخادم مزدحم حالياً، يرجى الانتظار قليلاً ثم المحاولة."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))


