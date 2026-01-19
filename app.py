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

# --- 2. تحميل المكتبة والبحث الاستكشافي ---
all_knowledge = []
def load_kb():
    global all_knowledge
    if os.path.exists("library_knowledge"):
        for f in sorted(os.listdir("library_knowledge")):
            if f.endswith(".json"):
                with open(os.path.join(path := "library_knowledge", f), "r", encoding="utf-8") as file:
                    all_knowledge.extend(json.load(file))
load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

def search_knowledge(query, top_k=3):
    q_norm = normalize(query)
    keywords = [w for w in q_norm.split() if len(w) > 2]
    scored = []
    for idx, u in enumerate(all_knowledge):
        content = normalize(u.get("content", ""))
        score = sum(15 for k in keywords if k in content) # رفع درجة التطابق
        if score > 0: scored.append((score, idx))
    scored.sort(reverse=True)
    
    indices = set()
    for _, idx in scored[:top_k]:
        # جلب سياق موسع لضمان النص الكامل (2 قبل و 20 بعد لضمان شمول القوائم)
        for i in range(max(0, idx-2), min(len(all_knowledge), idx+20)): 
            indices.add(i)
    return [all_knowledge[i] for i in sorted(list(indices))]

# --- 3. المعالجة والبرومبت "الصارم" ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_query = data.get("question")
        results = search_knowledge(user_query)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المادة في المكتبة."})

        # بناء السياق المرجعي بصورة واضحة
        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n### نص مرجعي رقم {i+1} ###\nالبيانات: {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\nالمحتوى الكامل:\n{r.get('content','')}\n---------------------------\n"

        prompt = f"""بصفتك باحثاً أكاديمياً رصيناً في اللسانيات العربية، مهمتك هي تقديم إجابة موثقة بناءً على النصوص المرفقة فقط.

        الأوامر الصارمة (نفذها دون أي تغيير):
        1. العبارة الاستهلالية: يجب أن تبدأ إجابتك حصراً بعبارة: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        
        2. منع التلخيص: ممنوع منعاً باتاً تلخيص المادة المرفقة. انقل "النصوص الكاملة" كما وردت في المتون المرفوعة.
        
        3. النسيج اللساني: قم بربط النصوص ببعضها لغوياً بكلمات رصينة (مثل: ويذهب البروفيسور في هذا الموضع إلى.. ، علاوة على تفصيل المبدأ القائل بـ..).
        
        4. التوثيق الحرفي: ضع كل فقرة منقولة حرفياً بين علامتي تنصيص "" متبوعة برقم المرجع [1]. يجب أن يكون الترقيم في المتن متسلسلاً تصاعدياً (1، 2، 3...) ولا تكرر الرقم أبداً؛ كل اقتباس له رقم جديد.
        
        5. الحاشية المنهجية: في نهاية إجابتك، اكتب عنواناً بارزاً باسم (المراجع:) ثم سرد بيانات المراجع المستخدمة في المتن فقط.
        
        6. اللغة: استخدم لغة عربية فصيحة فقط. ممنوع استخدام أي كلمات إنجليزية مثل "mention".

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
