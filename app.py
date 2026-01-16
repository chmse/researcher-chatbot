import os
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات جوجل Gemini (نسخة مستقرة جداً) ---
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# اختيار الاسم البرمجي المباشر والأكثر شهرة
MODEL_NAME = "gemini-1.5-flash"

# --- 2. تحميل المكتبة مع معالجة الأخطاء ---
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
                        if isinstance(data, list):
                            all_knowledge.extend(data)
                except Exception as e:
                    print(f"⚠️ خطأ في ملف {f_name}: {e}")
    print(f"📚 تم تحميل {len(all_knowledge)} وحدة.")

load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

# --- 3. محرك البحث ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"answer": "لم يصل سؤال صحيح."}), 400
        
        q = data["question"]
        print(f"❓ سؤال مستلم: {q}")

        # بحث سريع
        keywords = [w for w in normalize(q).split() if len(w) > 2]
        scored = []
        for i, u in enumerate(all_knowledge):
            content = normalize(u.get('content', ''))
            score = sum(3 for k in keywords if k in content)
            if score > 0: scored.append((score, i))
        
        scored.sort(reverse=True)
        # نأخذ أفضل 7 نتائج لضمان جلب كامل للمعلومات دون استهلاك ذاكرة
        top_results = [all_knowledge[i] for _, i in scored[:7]]

        if not top_results:
            return jsonify({"answer": "عذراً، لم أجد معلومات متعلقة بسؤالك في المكتبة المرفوعة."})

        # بناء السياق للذكاء الاصطناعي
        ctx = ""
        for i, r in enumerate(top_results):
            ctx += f"\n[مرجع:{i+1}] {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\nالنص: {r.get('content','')}\n"

        # التعليمات الصارمة (الدمج الأكاديمي والربط والنسخ)
        prompt = f"""أنت باحث أكاديمي متخصص في فكر الدكتور عبد الرحمن الحاج صالح. 
        ابدأ الإجابة بـ: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        
        التعليمات:
        1. انسخ النصوص المرفقة "حرفياً" وبالكامل كما وردت دون تغيير أو تلخيص.
        2. اربط بين النصوص بذكاء لغوي (وفي هذا السياق، كما يشير البروفيسور...).
        3. ضع النصوص المنقولة حرفياً بين "" متبوعة برقم المرجع المتسلسل [1]، [2] إلخ (لا تكرر الأرقام).
        4. المراجع في النهاية بدقة (المؤلف، الكتاب، الجزء، ص).
        
        المتــــون: {ctx}
        سؤال الباحث: {q}"""

        # محاولة توليد الإجابة
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        
        if response and response.text:
            return jsonify({"answer": response.text})
        else:
            return jsonify({"answer": "⚠️ اعتذر، لم يستطع المحرك صياغة إجابة الآن."})

    except Exception as e:
        print(f"❌ خطأ فادح: {str(e)}") # سيظهر هذا في سجلات Render لمعرفة السبب
        return jsonify({"answer": f"❌ حدث خطأ فني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
