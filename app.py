import os
import json
import re
import time
import traceback
import requests # مكتبة إضافية ضرورية للاتصال بمحركات خارجية مثل Groq
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات المفاتيح (جلب المفاتيح من Render) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- إعداد Gemini الأصلي ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_best_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if '1.5-flash' in m: return m
        return models[0] if models else "gemini-1.5-flash"
    except: return "gemini-1.5-flash"

MODEL_NAME = get_best_model()

# ================================================================
# الشرح: لكيفية إضافة API آخر (مثلاً OpenAI أو Claude):
# 1. أنشئ دالة جديدة تشبه 'call_groq' بالأسفل.
# 2. غير الرابط (URL) والموديل (Model Name) وحقول الـ JSON حسب متطلبات الموقع الجديد.
# 3. في الدالة 'ask' بالأسفل، أضف 'or call_new_api(prompt)' في سطر الاستجابة.
# ================================================================

# وظيفة محرك Gemini (المحرك الأول)
def call_gemini_engine(prompt, safety_settings):
    try:
        if not GEMINI_API_KEY: return None
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, safety_settings=safety_settings, generation_config={"temperature": 0.0})
        return response.text if response and response.text else None
    except: return None

# وظيفة محرك Groq (المحرك الاحتياطي)
def call_groq_engine(prompt):
    try:
        if not GROQ_API_KEY: return None
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile", # يمكنك تغيير الموديل هنا
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content']
    except: return None

# --- 2. تحميل المكتبة الكاملة (مع معالجة الأخطاء) ---
all_knowledge = []
def load_kb():
    global all_knowledge
    all_knowledge = []
    path = "library_knowledge"
    if os.path.exists(path):
        for f_name in sorted(os.listdir(path)):
            if f_name.endswith(".json"):
                try:
                    with open(os.path.join(path, f_name), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list): all_knowledge.extend(data)
                except Exception as e: print(f"⚠️ خطأ في {f_name}: {e}")
    print(f"📚 تم تحميل {len(all_knowledge)} وحدة معرفية.")

load_kb()

def normalize(t): 
    return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

# --- 3. محرك البحث الاستكشافي المطور ---
def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    keywords = [w for w in query_norm.split() if w not in {"ما","هي","أهم","مفهوم","في"} and len(w) > 2]
    
    scored = []
    for idx, unit in enumerate(units):
        content = normalize(unit.get("content", ""))
        score = sum(10 for kw in keywords if kw in content)
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', str(unit.get("content", "")).strip()):
            score += 5
        if score > 0: scored.append((score, idx))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    final_indices = set()
    for _, idx in scored[:top_k]:
        for i in range(max(0, idx-2), min(len(units), idx+15)):
            final_indices.add(i)
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال والتبديل التلقائي بين API ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data: return jsonify({"answer": "بيانات مفقودة"}), 400
        user_query = data.get("question")
        results = advanced_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد ذلك في المكتبة."})

        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n[م:{i+1}] {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\nالنص:{r.get('content','')}\n"

        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً رداً على سؤالكم:

        التعليمات الصارمة:
        1. ابدأ حصراً بعبارة الترحيب الأكاديمية المطلوبة.
        2. استخدم الربط اللغوي لتجعل الإجابة كتلة واحدة منسجمة.
        3. انسخ النصوص "حرفياً وبالكامل" بين علامتي تنصيص "" متبوعة بالمرجع [1] إلخ.
        4. ممنوع تكرار رقم المرجع.
        5. الحاشية في النهاية: (المراجع:) بدقة البيانات.
        6. الأمانة العلمية: ممنوع الحذف أو التلخيص.

        المواد المرجعية: {ctx}
        سؤال الباحث: {user_query}"""

        safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        # 🔄 تعدد محركات الـ API: نحاول مع Gemini، وإذا فشل ننتقل فوراً لـ Groq
        answer = call_gemini_engine(prompt, safety)
        
        if not answer:
            print("🔄 Gemini مزدحم أو فشل.. التبديل إلى Groq")
            answer = call_groq_engine(prompt)
        
        if answer:
            return jsonify({"answer": answer})
        return jsonify({"answer": "⚠️ نعتذر، جميع المحركات مشغولة حالياً."}), 500

    except Exception as e:
        print(f"❌ خطأ:\n{traceback.format_exc()}")
        return jsonify({"answer": f"❌ خطأ فني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
