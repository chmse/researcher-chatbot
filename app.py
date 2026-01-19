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

# --- الإعدادات ---
api_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

# --- محركات الاستجابة مع إعدادات تقليل الضغط ---
def call_gemini(prompt):
    try:
        if not api_key: return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        # إرسال طلب سريع بدون حرارة عالية
        res = model.generate_content(prompt, generation_config={"temperature": 0.1})
        return res.text
    except: return None

def call_groq(prompt):
    try:
        if not groq_key: return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
        res = requests.post(url, headers=headers, json=payload, timeout=50)
        return res.json()['choices'][0]['message']['content'] if res.status_code == 200 else None
    except: return None

# --- تحميل المكتبة والبحث الاستكشافي المتوازن ---
all_knowledge = []
def load_kb():
    global all_knowledge
    path = "library_knowledge"
    if os.path.exists(path):
        for f in sorted(os.listdir(path)):
            if f.endswith(".json"):
                with open(os.path.join(path, f), "r", encoding="utf-8") as file:
                    all_knowledge.extend(json.load(file))
load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

def search_optimized(query, units, top_k=2): # خفضنا k لتقليل الحِمل
    q_norm = normalize(query)
    keywords = [w for w in q_norm.split() if len(w) > 2]
    scored = []
    for idx, unit in enumerate(units):
        content = normalize(unit.get("content", ""))
        score = sum(15 for k in keywords if k in content)
        if score > 0: scored.append((score, idx))
    scored.sort(reverse=True)
    
    indices = set()
    for _, idx in scored[:top_k]:
        # سحب متوازن (15 فقرة تالية) يضمن المعلومة ويحمي الذاكرة
        for i in range(max(0, idx-2), min(len(units), idx+15)): 
            indices.add(i)
    return [units[i] for i in sorted(list(indices))]

# --- نقطة الاتصال مع صمام أمان للنصوص ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get("question")
        results = search_optimized(query, all_knowledge)
        
        if not results: return jsonify({"answer": "عذراً، لم أجد المادة في المكتبة."})

        # بناء السياق مع حد أقصى للكلمات (Safety Buffer)
        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n[المرجع_{i+1}]: {r.get('book','--')} ص:{r.get('page_pdf','--')}\n{r.get('content','')}\n"
            if len(ctx) > 10000: break # إذا تجاوز النص 10 آلاف حرف يتوقف السحب فوراً لمنع الانهيار

        prompt = f"""بصفتي باحثاً أكاديمياً، إليكم العرض الموثق رداً على: {query}
        المادة: {ctx}
        القواعد:
        1. ابدأ بعبارة الترحيب الأكاديمية (بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح...).
        2. انقل النصوص حرفياً وبالكامل بين "" مع الرقم [1]. 
        3. اربط بينها بأدوات ربط لغوية فقط (مثل: كما يشير في موضع آخر...).
        4. المراجع كاملة في الأسفل (بدون مراجع زائدة).
        """

        ans = call_gemini(prompt) or call_groq(prompt)
        
        if ans: return jsonify({"answer": ans})
        return jsonify({"answer": "⚠️ النظام متعب حالياً، حاول مرة أخرى."}), 500

    except:
        return jsonify({"answer": "❌ عذراً، تجاوز الطلب قدرة الذاكرة. حاول تضييق سؤالك."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
