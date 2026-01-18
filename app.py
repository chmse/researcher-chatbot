import os
import json
import re
import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات Gemini مع نظام الفشل الاحتياطي ---
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

MODELS = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-pro"]

def call_gemini(prompt):
    """محاولة الاتصال بالنماذج مع تجاهل فلاتر الحماية لتجنب رفض الطلبات الأكاديمية"""
    for m_name in MODELS:
        try:
            print(f"🔄 محاولة استخدام: {m_name}")
            model = genai.GenerativeModel(m_name)
            # تقليل حساسية الحماية لضمان نقل النصوص القديمة دون حظرها
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = model.generate_content(prompt, safety_settings=safety_settings)
            
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"⚠️ فشل الموديل {m_name}: {str(e)}")
            continue
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
                    all_knowledge.extend(json.load(f))
    print(f"📚 المكتبة جاهزة بـ {len(all_knowledge)} وحدة.")

load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

# --- 3. محرك البحث الاستكشافي ---
def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    keywords = [w for w in query_norm.split() if len(w) > 2]
    
    scored = []
    for idx, unit in enumerate(units):
        content = normalize(unit.get("content", ""))
        score = sum(5 for k in keywords if k in content)
        if score > 0: scored.append((score, idx))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    indices = set()
    for _, idx in scored[:top_k]:
        # جلب السياق الكامل
        for i in range(max(0, idx-2), min(len(units), idx+15)):
            indices.add(i)
                
    return [units[i] for i in sorted(list(indices))]

# --- 4. معالجة الطلبات ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data: return jsonify({"answer": "بيانات مفقودة"}), 400
        
        q = data.get("question")
        print(f"❓ السؤال: {q}")

        results = advanced_search(q, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد ذلك في المكتبة."})

        # بناء المادة المرجعية
        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n[م:{i+1}] {r.get('author','--')} | {r.get('book','--')} | ص:{r.get('page_pdf','--')}\nالنص:{r.get('content','')}\n"

        prompt = f"""أنت باحث أكاديمي متخصص في فكر الدكتور عبد الرحمن الحاج صالح. 
        ابدأ حصراً بعبارة الترحيب الأكاديمية: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        
        التعليمات:
        1. انسخ النصوص المرفقة حرفياً وبالكامل داخل "" متبوعة بالمرجع [1]، [2] إلخ.
        2. اربط بينها بلغة رصينة دون تأويل شخصي.
        3. اذكر المراجع في النهاية بالترتيب.

        المتون: {ctx}
        سؤال الباحث: {q}"""

        answer = call_gemini(prompt)
        
        if answer:
            return jsonify({"answer": answer})
        else:
            return jsonify({"answer": "⚠️ اعتذر، لم يتمكن المحرك من الوصول للنص (قد يكون مفتاح API محظور أو تحت المراجعة)."}), 500

    except Exception as e:
        print(f"❌ خطأ فادح:\n{traceback.format_exc()}")
        return jsonify({"answer": f"❌ حدث خطأ فني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
