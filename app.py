import os
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.api_core import exceptions

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات جوجل Gemini واكتشاف النموذج ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_model_name():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if '1.5-flash' in m: return m
        return models[0] if models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

MODEL_NAME = get_model_name()

# --- 2. تحميل المكتبة الكاملة من المجلد ---
all_knowledge = []
KB_PATH = "library_knowledge"

def load_library():
    global all_knowledge
    all_knowledge = []
    if os.path.exists(KB_PATH):
        # ترتيب الملفات لضمان تسلسل الأفكار
        for filename in sorted(os.listdir(KB_PATH)):
            if filename.endswith(".json"):
                with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
    return len(all_knowledge)

load_library()

# --- 3. محرك البحث الاستكشافي (لجلب القوائم والتعليل) ---
def normalize(text):
    if not text: return ""
    return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", text))).strip()

def advanced_search(query, units, top_k=3):
    """خوارزمية تبحث عن طرف الخيط ثم تسحب كل العناصر المرتبطة تلو بعضها"""
    query_norm = normalize(query)
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        content = unit.get("content", "")
        score = sum(5 for kw in keywords if kw in normalize(content))
        # إعطاء أفضلية للمقاطع التي تبدأ بترقيم
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', content.strip()): score += 2
        
        if score > 0:
            scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    # جلب المقطع المختار + مسح تتبعي عميق (15 وحدة) لضمان جلب القوائم كاملة
    for _, idx in scored_indices[:top_k]:
        # نأخذ مقطعين قبل (للتمهيد) و15 مقطع بعد (لسحب القائمة كاملة)
        for i in range(max(0, idx-2), min(len(units), idx+15)):
            u_content = units[i].get("content", "")
            # ضم المقطع إذا كان مرتبطاً بالترقيم أو يحتوي على كلمات بحث
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()) or any(k in normalize(u_content) for k in keywords):
                final_indices.add(i)
            # توقف إذا ابتعدنا عن السياق ولم نعد نجد ترقيماً
            if i > idx + 5 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()):
                break
                
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال (المصفي والموثق الأكاديمي) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        results = advanced_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n[مرجع رقم: {i+1}] [ص: {u.get('page_pdf','--')}] [كتاب: {u.get('book','--')}] [مؤلف: {u.get('author','--')}] [ج: {u.get('part','--')}]\n{u['content']}\n"
        
        # الموجه (Prompt) المطور حسب طلبك
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، قمت باستخراج المفاهيم والفقرات المرقمة من النصوص المرفقة بدقة متناهية، رداً على سؤالكم:

        مهمتك صياغة إجابة 'مدمجة' و 'مرتبة' وفق الشروط التالية:
        1. الالتزام بالنص: انقل المعلومات من الوحدات المرفقة فقط. يمكنك تحسين الصياغة اللغوية والإملائية دون تغيير المعنى أو الألفاظ اللسانية للمؤلف.
        2. هيكل الفقرات: ابدأ كل نقطة أو عنصر مرقم (1، 2، 3...) في سطر جديد تماماً.
        3. التوثيق (المتن): ضع النص المقتبس بين علامتي تنصيص ' ' متبوعاً برقم المرجع المذكور في السياق [1].
        4. التوثيق (الحاشية): في نهاية الإجابة، اكتب عنواناً بارزاً (المراجع:) ثم سرد المراجع بالصيغة: رقم المرجع- اسم المؤلف، اسم الكتاب، الجزء، ص: رقم الصفحة.
        5. الصرامة: لا تضف أي معلومات خارجية. التوسع يكون فقط من خلال ما ورد في النصوص المرفقة.
        6. الربط: استخدم روابط لغوية محكمة للجمع بين الأفكار لتظهر كإجابة متصلة ومنسجمة.

        النصوص المرجعية المستخرجة من المكتبة:
        {ctx_text}

        سؤال المستخدم:
        {user_query}
        """

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        # محاولة التوليد مع معالجة الازدحام
        for _ in range(3):
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(12)
        
        return jsonify({"answer": "⚠️ الخادم مزدحم حالياً، يرجى المحاولة مرة أخرى الآن."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
