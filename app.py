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

# --- 1. الإعدادات ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- محرك Gemini ---
def call_gemini(prompt):
    try:
        if not GEMINI_API_KEY: return None
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        return response.text if response else None
    except: return None

# --- محرك Groq ---
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
    keywords = [w for w in query_norm.split() if len(w) > 2]
    
    scored = []
    for idx, unit in enumerate(units):
        content = normalize(unit.get("content", ""))
        score = sum(10 for k in keywords if k in content) # نقاط للكلمات المفتاحية
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', str(unit.get("content", "")).strip()):
            score += 5 # نقاط إضافية إذا بدأ بترقيم
        if score > 0: scored.append((score, idx))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    for _, idx in scored[:top_k]:
        # سحب السياق (2 قبل و 15 بعد) لضمان النسخ الكامل للمعلومة
        for i in range(max(0, idx-2), min(len(units), idx+15)):
            final_indices.add(i)
                
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال والبرومبت المدمج ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data: return jsonify({"answer": "بيانات مفقودة"}), 400
        
        user_query = data.get("question")
        print(f"❓ استفسار: {user_query}")

        results = advanced_search(user_query, all_knowledge)
        if not results:
            return jsonify({"answer": "عذراً، لم أجد ذلك في المكتبة."})

        # بناء المادة المرجعية الموثقة
        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n[م:{i+1}] {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\nالنص:{r.get('content','')}\n"

        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً رداً على سؤالكم:

        التعليمات الصارمة:
        1. ابدأ حصراً بعبارة الترحيب الأكاديمية المطلوبة.
        2. استخدم الربط اللغوي (وفي هذا الصدد، علاوة على ما سبق، يوضح البروفيسور...) لتجعل الإجابة كتلة واحدة منسجمة.
        3. عند الوصول للمعلومة المأخوذة من المكتبة، انسخها "حرفياً وبالكامل" وضعها بين علامتي تنصيص "" متبوعة بالمرجع المتسلسل [1]، [2] إلخ.
        4. ممنوع تكرار رقم المرجع؛ كل اقتباس جديد يأخذ رقماً جديداً حتى لو كان من نفس الصفحة.
        5. الحاشية في النهاية: (المراجع:) ثم سرد المراجع بنفس الأرقام بالصيغة: الرقم- المؤلف، الكتاب، الجزء، ص: رقم الصفحة.
        6. الأمانة العلمية: ممنوع الحذف أو التلخيص، انقل النصوص كما وردت كاملة وبصرامة.


        المادة العلمية المرجعية المرفقة للنسخ:
        {ctx}

        سؤال الباحث للرد عليه:
        {user_query}
        """

        print("🔍 إرسال الطلب للنظام المزدوج...")
        ans = call_gemini(prompt)
        if not ans:
            print("🔄 التبديل للمحرك الاحتياطي...")
            ans = call_groq(prompt)

        if ans: return jsonify({"answer": ans})
        return jsonify({"answer": "❌ الخادم مزدحم، حاول مجدداً."}), 500

    except Exception:
        print(traceback.format_exc())
        return jsonify({"answer": "❌ حدث خطأ فني."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)



