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

# --- 1. إعدادات جوجل Gemini ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_model_name():
    try:
        # فحص النماذج المتاحة لتجنب خطأ 404
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if '1.5-flash' in m: return m
        return models[0]
    except:
        return "models/gemini-1.5-flash"

MODEL_NAME = get_model_name()

# --- 2. تحميل المكتبة ---
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

# --- 3. محرك البحث الموسع (سريع وذكي) ---
def normalize(text):
    if not text: return ""
    return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", text))).strip()

def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        content = unit.get("content", "")
        score = sum(2 for kw in keywords if kw in normalize(content))
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', content.strip()): score += 1
        if score > 0:
            score += (1 - (unit.get("page_pdf", 0) / 500))
            scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    # جلب المقطع المختار + 10 مقاطع تالية لضمان شمولية القوائم
    for _, idx in scored_indices[:top_k]:
        for i in range(max(0, idx-1), min(len(units), idx+10)):
            content = units[i].get("content", "")
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', content.strip()):
                final_indices.add(i)
            if i > idx + 4 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', content.strip()):
                break
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال (API) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        results = advanced_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n[مرجع: {i+1}] [ص: {u.get('page_pdf','--')}] [كتاب: {u.get('book','--')}] [مؤلف: {u.get('author','--')}] [ج: {u.get('part','--')}]\n{u['content']}\n"
        
        prompt = f"""أنت باحث أكاديمي متخصص في فكر الحاج صالح. استخرج كل المفاهيم والفقرات المرقمة (1، 2، 3...) من النصوص المرفقة حصراً.
        القواعد:
        1. المتن: انقل النص حرفياً بين علامتي ' ' متبوعاً برقم المرجع [1].
        2. الحاشية: في النهاية اكتب (المراجع:) ثم: رقم- المؤلف، الكتاب، الجزء، ص: الصفحة.
        3. الصرامة: انقل الجمل كما هي داخل علامتي تنصيص.
        
        النصوص:\n{ctx_text}\nسؤال المستخدم: {user_query}"""

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        for _ in range(3):
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(10)
        
        return jsonify({"answer": "⚠️ ضغط كبير على الخادم، يرجى المحاولة بعد دقيقة."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
