
import os
import json
import re
import time
import requests
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات المفاتيح (Environment Groups: chmsxp) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- محرك 1: Gemini ---
def call_gemini(prompt):
    try:
        if not GEMINI_API_KEY: return None
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        return response.text if response else None
    except: return None

# --- محرك 2: Groq (الاحتياطي السريع) ---
def call_groq(prompt):
    try:
        if not GROQ_API_KEY: return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        res = requests.post(url, headers=headers, json=payload, timeout=40)
        return res.json()['choices'][0]['message']['content'] if res.status_code == 200 else None
    except: return None

# --- تحميل المكتبة والبحث ---
all_knowledge = []
def load_kb():
    global all_knowledge
    if os.path.exists("library_knowledge"):
        for f in sorted(os.listdir("library_knowledge")):
            if f.endswith(".json"):
                with open(os.path.join("library_knowledge", f), "r", encoding="utf-8") as file:
                    all_knowledge.extend(json.load(file))
load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

def search_knowledge(query, top_k=3):
    q_norm = normalize(query)
    keywords = [w for w in q_norm.split() if len(w) > 2]
    scored = []
    for idx, u in enumerate(all_knowledge):
        content = normalize(u.get("content", ""))
        score = sum(10 for k in keywords if k in content)
        if score > 0: scored.append((score, idx))
    scored.sort(reverse=True)
    
    indices = set()
    for _, idx in scored[:top_k]:
        for i in range(max(0, idx-2), min(len(all_knowledge), idx+15)): indices.add(i)
    return [all_knowledge[i] for i in sorted(list(indices))]

# --- 4. معالجة الطلبات بالمنطق الأكاديمي الصارم ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_query = data.get("question")
        results = search_knowledge(user_query)
        if not results: return jsonify({"answer": "عذراً، لم أجد المعلومة."})

        # إرسال البيانات المرجعية للذكاء الاصطناعي مع أرقام معرفية بسيطة
        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n[نص_{i+1}]: {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\nمحتوى:{r.get('content','')}\n"

        prompt = f"""بصفتك باحثاً أكاديمياً متخصصاً، صِغ إجابة موثقة رداً على سؤال الباحث وفقاً للمعطيات التالية:
        
        سؤال الباحث: {user_query}
        المادة العلمية المتاحة: {ctx}

        أوامر التنسيق (صارمة جداً):
        1. ابدأ بعبارة الترحيب الأكاديمية المطلوبة.
        2. ادمج بين "النقل الحرفي للنصوص" و "روابط الربط اللغوية الذكية".
        3. ضع كل اقتباس حرفي بين علامتي "" متبوعاً برقم [1]، [2]... بشكل متسلسل وتصاعدي.
        4. ممنوع القفز في الأرقام أو تكرارها في المتن.
        5. قائمة المراجع في النهاية: لا تذكر إلا المراجع التي قمت باستخدامها فعلياً داخل إجابتك. (احذف أي مراجع زائدة).
        6. النسخ يجب أن يكون كاملاً للفقرة دون اختصار مخِل.
        """

        # تنفيذ المحاولة المزدوجة
        print("🔍 محاولة الرد (Gemini)...")
        ans = call_gemini(prompt)
        if not ans:
            print("🔄 محاولة التبديل (Groq)...")
            ans = call_groq(prompt)

        if ans: return jsonify({"answer": ans})
        return jsonify({"answer": "❌ المحركات مزدحمة."}), 500

    except Exception:
        print(traceback.format_exc())
        return jsonify({"answer": "❌ خطأ فني."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
