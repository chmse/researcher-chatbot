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

        # prompt = f"""بصفتك باحثاً أكاديمياً رصيناً في اللسانيات العربية، مهمتك هي تقديم إجابة موثقة بناءً على النصوص المرفقة فقط.

        # الموجه (Prompt) الجديد: صرامة النسخ + دقة التوثيق المتسلسل
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً رداً على سؤالكم:

        مهمتك صياغة إجابة 'كاملة' و 'مترابطة' وفق الشروط الصارمة التالية:
        1. العبارة الاستهلالية: ابدأ حصراً بـ: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        2. النقل والنسخ: انقل النصوص المرجعية 'حرفياً وبالكامل' دون أي تلخيص أو حذف للجمل الطويلة. استخدم روابط لغوية ذكية للربط بين هذه النصوص (مثل: كما يشير في موضع آخر، علاوة على ذلك يقرر...).
        3. التوثيق المتسلسل في المتن: ضع كل اقتباس حرفي بين علامتي تنصيص "" متبوعاً برقم مرجع متسلسل [1]، ثم [2] وهكذا.
        4. ملاحظة: يجب أن يكون الترقيم متسلسلاً (1، 2، 3...) حسب ظهوره في الإجابة، ولا تكرر نفس الرقم أبداً؛ كل اقتباس جديد يأخذ رقماً جديداً حتى لو كان من نفس الصفحة.
        5. المراجع (الحاشية): في نهاية الإجابة، اكتب عنواناً بارزاً (المراجع:) ثم سرد المراجع المقابلة لكل رقم استخدمته بالصيغة: رقم المرجع- اسم المؤلف، اسم الكتاب، الجزء، ص: الصفحة.
        6. الصرامة: لا تضف أي معلومة خارجية أو تأويل. انقل القوائم المرقمة كما وردت تماماً.
        7. اللغة: استخدم لغة عربية فصيحة فقط. ممنوع استخدام أي كلمات إنجليزية مثل "mention".

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

