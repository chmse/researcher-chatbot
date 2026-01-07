import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.api_core import exceptions

app = Flask(__name__)
CORS(app)

# --- 1. إعدادات جوجل Gemini ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_model():
    try:
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

MODEL_NAME = get_model()

# --- 2. تحميل المكتبة ---
all_knowledge = []
KB_PATH = "library_knowledge"

def load_library():
    global all_knowledge
    all_knowledge = []
    if os.path.exists(KB_PATH):
        for filename in os.listdir(KB_PATH):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_knowledge.extend(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return len(all_knowledge)

load_library()

# --- 3. خوارزمية البحث ---
STOP_WORDS = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي"}
def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", t))).strip()
def stem(w): 
    for s in ["ات","ون","ين","هم","نا"]: 
        if w.endswith(s) and len(w) > 4: return w[:-len(s)]
    return w

def smart_search(query, top_k=2):
    q_norm = normalize(query)
    keywords = {stem(w) for w in q_norm.split() if w not in STOP_WORDS and len(w) > 2}
    
    scored = []
    for idx, u in enumerate(all_knowledge):
        content_norm = normalize(u.get("content", ""))
        score = sum(1 for kw in keywords if kw in content_norm)
        if score > 0: scored.append((score, idx))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    indices = set()
    for _, idx in scored[:top_k]:
        indices.update({max(0, idx-1), idx, min(len(all_knowledge)-1, idx+1)})
    return [all_knowledge[i] for i in sorted(list(indices))]

# --- 4. نقطة الاتصال (API Endpoint) ---
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_query = data.get("question")
    
    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    results = smart_search(user_query)
    if not results:
        return jsonify({"answer": "عذراً، لم أجد معلومة في المكتبة تتعلق بسؤالك."})

    # بناء نص السياق مع معالجة الخانات المفقودة (حل مشكلة KeyError)
    ctx_parts = []
    for i, u in enumerate(results):
        ref_num = i + 1
        book = u.get('book', 'غير مذكور')
        author = u.get('author', 'غير مذكور')
        page = u.get('page_pdf', 'غ/م')
        part = u.get('part', '1') # الافتراضي الجزء الأول إذا لم يوجد
        content = u.get('content', '')
        
        ctx_parts.append(f"\n[مرجع: {ref_num}] [كتاب: {book}] [مؤلف: {author}] [ج: {part}] [ص: {page}]\n{content}\n")

    ctx_text = "".join(ctx_parts)
    
    prompt = f"""أنت محقق أكاديمي ملتزم بالنقل الحرفي الصارم من النصوص المرفقة فقط.
    السياق المرجعي: {ctx_text}
    سؤال الباحث: {user_query}
    التعليمات:
    1. المتن: انقل النص حرفياً بين علامتي تنصيص ' ' متبوعاً برقم المرجع [1].
    2. الحاشية: في النهاية، اكتب عنوان (المراجع:) ثم البيانات لكل رقم.
    3. الصرامة: لا تضف أي معلومة خارجية."""

    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"❌ حدث خطأ: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
