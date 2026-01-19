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

# --- 1. إعدادات الوصول للمحركات (Gemini + Groq) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def call_gemini(prompt):
    try:
        if not GEMINI_API_KEY: return None
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        res = model.generate_content(prompt, generation_config={"temperature": 0.0})
        return res.text if res else None
    except: return None

def call_groq(prompt):
    try:
        if not GROQ_API_KEY: return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        res = requests.post(url, headers=headers, json=payload, timeout=50)
        return res.json()['choices'][0]['message']['content'] if res.status_code == 200 else None
    except: return None

# --- 2. محرك البحث الموسع (Deep Context Retrieval) ---
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

def search_deep_context(query, top_k=3):
    q_norm = normalize(query)
    keywords = [w for w in q_norm.split() if len(w) > 2]
    scored = []
    for idx, unit in enumerate(all_knowledge):
        content = normalize(unit.get("content", ""))
        score = sum(20 for k in keywords if k in content) # نقاط عالية للدقة
        if score > 0: scored.append((score, idx))
    scored.sort(reverse=True)
    
    indices = set()
    for _, idx in scored[:top_k]:
        # سحب فقرتين قبل و 25 فقرة بعد (لسحب فصول فرعية كاملة أحياناً)
        for i in range(max(0, idx-2), min(len(all_knowledge), idx+25)): 
            indices.add(i)
    return [all_knowledge[i] for i in sorted(list(indices))]

# --- 3. المدير الأكاديمي وصياغة البرومبت النهائي ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get("question")
        results = search_deep_context(query)
        
        if not results:
            return jsonify({"answer": "عذراً، لم أجد المادة العلمية المتعلقة بهذا السؤال."})

        # دمج المراجع للنموذج
        full_context = ""
        for i, r in enumerate(results):
            full_context += f"\n[المرجع_{i+1}]: {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\n{r.get('content','')}\n"

        prompt = f"""بصفتك باحثاً أكاديمياً ملتزماً بالصرامة العلمية في فكر الدكتور عبد الرحمن الحاج صالح، صغ إجابة 'منقولة' و'مرتبة' وفقاً للأوامر التالية:

        سؤال الباحث: {query}

        النصوص العلمية المستخرجة (المتن):
        {full_context}

        قوانين التنفيذ (عدم التنفيذ يعني فشل المهمة):
        1. العبارة الافتتاحية: ابدأ بعبارة: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        2. النقل الحرفي الصارم: يُحظر التلخيص أو الشرح أو التأويل. انقل الفقرات من 'المتن المرفق' كاملةً كما هي بين علامتي تنصيص "" متبوعة برقم متسلسل [1].
        3. الربط اللساني فقط: استخدم كلمات ربط لسانية (مثل: وبناءً عليه، علاوة على ما ذكره البروفيسور، كما يتضح في قوله...) لنسج الفقرات المنقولة في وحدة واحدة، دون التدخل في معناها.
        4. المراجع الحصرية: في النهاية اذكر عنوان (المراجع:) ثم سرد المراجع التي تم اقتباسها فعلياً فقط، مع مطابقة الأرقام المتسلسلة (1، 2، 3...).
        5. التفصيل: انقل القوائم، التعريفات، والتحليلات كاملةً ولا تحذف منها حرفاً واحداً.
        """

        # التشغيل المتبادل
        print("🔍 استلام طلب جديد - محاولة Gemini...")
        ans = call_gemini(prompt)
        if not ans:
            print("🔄 Gemini غير متاح - التبديل الفوري لـ Groq...")
            ans = call_groq(prompt)

        if ans: return jsonify({"answer": ans})
        return jsonify({"answer": "❌ نأسف، الأنظمة مشغولة حالياً."}), 500

    except Exception:
        print(traceback.format_exc())
        return jsonify({"answer": "❌ حدث خطأ فني غير متوقع."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
