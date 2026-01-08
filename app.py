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

def get_model_name():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if '1.5-flash' in m: return m
        return models[0] if models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

MODEL_NAME = get_model_name()

# --- 2. تحميل المكتبة الكاملة ---
all_knowledge = []
KB_PATH = "library_knowledge"

def load_library():
    global all_knowledge
    all_knowledge = []
    if os.path.exists(KB_PATH):
        for filename in sorted(os.listdir(KB_PATH)):
            if filename.endswith(".json"):
                with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
    return len(all_knowledge)

load_library()

# --- 3. محرك البحث الذكي الاستكشافي ---
def normalize(text):
    if not text: return ""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("[ةه]", "ه", text)
    text = re.sub("ى", "ي", text)
    return re.sub(r'[\u064B-\u0652]', '', text).strip()

def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        content_norm = normalize(unit.get("content", ""))
        score = sum(5 for kw in keywords if kw in content_norm)
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', unit.get("content", "").strip()): score += 2
        if score > 0:
            score += (1 - (unit.get("page_pdf", 0) / 1000))
            scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    for _, idx in scored_indices[:top_k]:
        # جلب المقطع مع مسح تتابعي لضمان الشمولية (15 وحدة تالية)
        for i in range(max(0, idx-1), min(len(units), idx+15)):
            u_content = units[i].get("content", "")
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()) or any(k in normalize(u_content) for k in keywords):
                final_indices.add(i)
            if i > idx + 5 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()):
                break
                
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال (المصفي والموثق الأكاديمي الصارم) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        results = advanced_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        # بناء سياق النصوص المرجعية
        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n--- [معرف المرجع: {i+1}] ---\nالمؤلف: {u.get('author','--')} | الكتاب: {u.get('book','--')} | ج: {u.get('part','--')} | ص: {u.get('page_pdf','--')}\nالنص: {u['content']}\n"
        
        # الموجه (Prompt) المطور بالأدوات الأكاديمية المطلوبة
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
        
        # محاولة التوليد مع معالجة الازدحام
        for _ in range(3):
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(15)
        
        return jsonify({"answer": "⚠️ الخادم مزدحم حالياً، يرجى المحاولة مرة أخرى بعد ثوانٍ."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
