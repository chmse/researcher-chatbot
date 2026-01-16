import os
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات جوجل Gemini وحل مشكلة الـ 404 ---
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# قائمة بالأسماء التقنية المحتملة للنموذج (سنجربها بالترتيب)
CANDIDATE_MODELS = [
    "models/gemini-1.5-flash",
    "gemini-1.5-flash",
    "models/gemini-pro",
    "gemini-pro"
]

def try_generate_content(prompt):
    """دالة ذكية تجرب كل الأسماء المتاحة للنموذج حتى تنجح"""
    for model_name in CANDIDATE_MODELS:
        try:
            print(f"🔄 محاولة استخدام النموذج: {model_name}")
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(prompt)
            if response:
                return response.text
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e).lower():
                print(f"❌ الاسم {model_name} غير مدعوم، ننتقل للتالي...")
                continue
            else:
                # إذا كان الخطأ ليس 404 (مثل ضغط الخادم)، انتظر ثوانٍ
                print(f"⚠️ خطأ مؤقت: {e}")
                time.sleep(2)
    return None

# --- 2. تحميل المكتبة ---
all_knowledge = []
def load_kb():
    global all_knowledge
    all_knowledge = []
    path = "library_knowledge"
    if os.path.exists(path):
        for f_name in sorted(os.listdir(path)):
            if f_name.endswith(".json"):
                with open(os.path.join(path, f_name), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list): all_knowledge.extend(data)
    print(f"📚 تم تحميل {len(all_knowledge)} وحدة.")

load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

# --- 3. محرك البحث ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        q = data.get("question")
        if not q: return jsonify({"answer": "لم يصل سؤال"}), 400
        
        # البحث في النصوص
        keywords = [w for w in normalize(q).split() if len(w) > 2]
        scored = []
        for i, u in enumerate(all_knowledge):
            content = normalize(u.get('content', ''))
            score = sum(3 for k in keywords if k in content)
            if score > 0: scored.append((score, i))
        
        scored.sort(reverse=True)
        top_results = [all_knowledge[i] for _, i in scored[:8]]

        if not top_results:
            return jsonify({"answer": "عذراً، لم أجد معلومات متعلقة بسؤالك."})

        # بناء السياق
        ctx = ""
        for i, r in enumerate(top_results):
            ctx += f"\n[مرجع:{i+1}] {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\nالنص: {r.get('content','')}\n"

        # صياغة التعليمات (البرومبت)
        prompt = f"""أنت باحث أكاديمي متخصص في فكر الدكتور عبد الرحمن الحاج صالح. 
        ابدأ الإجابة بعبارة الترحيب الأكاديمية الصارمة.
        - انسخ النصوص المرفقة حرفياً وبالكامل بين "" مع رقم المرجع [1].
        - اربط الأفكار بذكاء وتجنب تكرار أرقام المراجع.
        - الحاشية (المراجع) في الأسفل كاملة البيانات.
        
        النصوص المستخرجة: {ctx}
        سؤال الباحث: {q}"""

        # طلب الإجابة من الدالة الذكية (Fallback Logic)
        final_answer = try_generate_content(prompt)
        
        if final_answer:
            return jsonify({"answer": final_answer})
        else:
            return jsonify({"answer": "❌ عذراً، فشل الاتصال بمحركات الذكاء الاصطناعي. يرجى مراجعة مفتاح API."}), 500

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ داخلي: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
