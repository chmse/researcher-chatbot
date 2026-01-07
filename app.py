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

def get_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if '1.5-flash' in m.name: return m.name
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
                with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
    return len(all_knowledge)

load_library()

# --- 3. خوارزمية البحث ---
STOP_WORDS = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي"}
SYNONYMS = {"توثيق": ["حجج", "إسناد", "رواية"], "أصل": ["أساس", "قاعدة", "منطلق"]}

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", t))).strip()
def stem(w): 
    for s in ["ات","ون","ين","هم","نا"]: 
        if w.endswith(s) and len(w) > 4: return w[:-len(s)]
    return w

def smart_search(query, top_k=2):
    q_norm = normalize(query)
    keywords = {stem(w) for w in q_norm.split() if w not in STOP_WORDS and len(w) > 2}
    for k in list(keywords):
        if k in SYNONYMS: keywords.update(SYNONYMS[k])
    
    scored = []
    for idx, u in enumerate(all_knowledge):
        content_norm = normalize(u.get("content", ""))
        score = sum(1 for kw in keywords if kw in content_norm)
        if score > 0: scored.append((score, idx))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    indices = set()
    for _, idx in scored[:top_k]:
        indices.update({idx, max(0, idx-1), min(len(all_knowledge)-1, idx+1)})
    return [all_knowledge[i] for i in sorted(list(indices))]

# --- 4. نقطة الاتصال المصلحة ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        
        if not user_query:
            return jsonify({"error": "No question"}), 400

        results = smart_search(user_query)
        if not results:
            return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في كتب المكتبة."})

        # تم تعديل السطر أدناه لاستخدام .get() لتجنب خطأ KeyError
        ctx_text = ""
        for i, u in enumerate(results):
            page = u.get('page_pdf', '--')
            book = u.get('book', 'كتاب غير محدد')
            author = u.get('author', 'غير معروف')
            part = u.get('part', '--')
            content = u.get('content', '')
            ctx_text += f"\n[مرجع: {i+1}] [ص: {page}] [كتاب: {book}] [مؤلف: {author}] [ج: {part}]\n{content}\n"
        
        prompt = f"""أنت محقق أكاديمي ملتزم بالنقل الحرفي الصارم من النصوص المرفقة فقط.
        السياق المرجعي: {ctx_text}
        سؤال الباحث: {user_query}
        التعليمات:
        1. المتن: انقل النص حرفياً بين علامتي تنصيص ' ' متبوعاً برقم المرجع [1].
        2. الحاشية: في النهاية اذكر المراجع: رقم- المؤلف، الكتاب، الجزء، ص: الصفحة.
        3. الصرامة: لا تضف أي شرح خارجي."""

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        return jsonify({"answer": response.text})

    except exceptions.TooManyRequests:
        return jsonify({"answer": "⚠️ الخادم مزدحم، يرجى المحاولة بعد قليل."})
    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
