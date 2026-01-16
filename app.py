import os
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

# إعدادات Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_model_name():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if '1.5-flash' in m: return m
        return "gemini-1.5-flash"
    except: return "gemini-1.5-flash"

MODEL_NAME = get_model_name()

# تحميل المكتبة (مرة واحدة فقط)
all_knowledge = []
def load_kb():
    global all_knowledge
    if os.path.exists("library_knowledge"):
        for f_name in sorted(os.listdir("library_knowledge")):
            if f_name.endswith(".json"):
                with open(os.path.join("library_knowledge", f_name), "r", encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t)))).strip()

@app.route('/ask', methods=['POST'])
def ask():
    try:
        q = request.json.get("question")
        if not q: return jsonify({"answer": "سؤال فارغ"}), 400

        # بحث سريع جداً لتقليل الضغط
        keywords = [w for w in normalize(q).split() if len(w) > 2]
        scored = []
        for i, u in enumerate(all_knowledge):
            score = sum(1 for k in keywords if k in normalize(u.get('content', '')))
            if score > 0: scored.append((score, i))
        
        scored.sort(reverse=True)
        results = [all_knowledge[i] for _, i in scored[:5]] # جلب أفضل 5 نتائج فقط لتوفير الذاكرة

        if not results: return jsonify({"answer": "لم أجد معلومة بدقة."})

        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n[م:{i+1}] {r.get('author')} | {r.get('book')} | ج:{r.get('part')} | ص:{r.get('page_pdf')}\nالنص: {r.get('content')}\n"

        prompt = f"""بصفتي باحثاً أكاديمياً، إليك الإجابة الموثقة:
        ابدأ بعبارة: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً رداً على سؤالكم:"
        - انسخ النصوص المرفقة حرفياً داخل "" متبوعة برقم المرجع [1] وتجنب تكرار الأرقام.
        - اربط الأفكار بذكاء.
        - المراجع في النهاية بالترتيب.
        
        النصوص: {ctx}
        سؤال الباحث: {q}"""

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return jsonify({"answer": response.text})

    except Exception as e: return jsonify({"answer": f"خطأ: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
