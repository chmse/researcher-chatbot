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
    """البحث عن اسم النموذج الصحيح المتاح في الحساب"""
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

# --- 3. خوارزمية البحث الذكي (الجذور + المترادفات + الجوار) ---
STOP_WORDS = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي"}
SYNONYMS = {"توثيق": ["حجج", "إسناد", "رواية"], "أصل": ["قاعدة", "أساس", "منطلق"]}

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", t))).strip()
def stem(w): 
    for s in ["ات","ون","ين","هم","نا"]: 
        if w.endswith(s) and len(w) > 4: return w[:-len(s)]
    return w

def library_smart_search(query, top_k=2):
    q_norm = normalize(query)
    words = [stem(w) for w in q_norm.split() if w not in STOP_WORDS and len(w) > 2]
    expanded_keywords = set(words)
    for w in words:
        if w in SYNONYMS: expanded_keywords.update(SYNONYMS[w])
    
    scored_indices = []
    for idx, u in enumerate(all_knowledge):
        content_norm = normalize(u.get("content", ""))
        score = sum(1 for kw in expanded_keywords if kw in content_norm)
        if score > 0:
            score += (1 - (u.get("page_pdf", 0) / 500))
            scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    for _, idx in scored_indices[:top_k]:
        # جلب المقطع مع جواره لضمان التعليل
        final_indices.update({max(0, idx-1), idx, min(len(all_knowledge)-1, idx+1)})
    
    return [all_knowledge[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال (API Endpoint) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"error": "No question"}), 400

        results = library_smart_search(user_query)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        # بناء نص السياق الموثق
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
        2. الحاشية: في النهاية، اذكر المراجع بالصيغة: رقم- المؤلف، الكتاب، الجزء، ص: الصفحة.
        3. الصرامة: لا تضف أي شرح خارجي من عندك نهائياً."""

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        # معالجة ضغط الطلبات (Retry Loop)
        for attempt in range(3):
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(10)
        
        return jsonify({"answer": "⚠️ الخادم مزدحم حالياً، يرجى إعادة المحاولة بعد ثوانٍ."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

