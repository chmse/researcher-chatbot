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

# --- 1. إعدادات جوجل Gemini واكتشاف النموذج (هذا الجزء هو سر نجاح الاتصال) ---
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
        for filename in sorted(os.listdir(KB_PATH)):
            if filename.endswith(".json"):
                with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
    return len(all_knowledge)

load_library()

# --- 3. محرك البحث الاستكشافي (هذا الجزء يضمن سحب القوائم والتعليلات كاملة) ---
def normalize(text):
    if not text: return ""
    return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", text))).strip()

def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        content = unit.get("content", "")
        score = sum(5 for kw in keywords if kw in normalize(content))
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', content.strip()): score += 2
        if score > 0:
            scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    # جلب سياق موسع (2 قبل و 15 بعد) لضمان جلب كامل الفقرة
    for _, idx in scored_indices[:top_k]:
        for i in range(max(0, idx-2), min(len(units), idx+15)):
            u_content = units[i].get("content", "")
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()) or any(k in normalize(u_content) for k in keywords):
                final_indices.add(i)
            if i > idx + 7 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()):
                break
                
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال (تحديث الموجه ليكون صارماً وأكاديمياً) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        results = advanced_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة المرفوعة."})

        # بناء نصوص السياق بدقة
        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n--- [معرف السياق: {i+1}] ---\nالمؤلف: {u.get('author','--')} | الكتاب: {u.get('book','--')} | ج: {u.get('part','--')} | ص: {u.get('page_pdf','--')}\nالنص: {u['content']}\n"
        
        # الموجه (Prompt) الجديد: صرامة النسخ + دقة التوثيق المتسلسل
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً رداً على سؤالكم:

        مهمتك صياغة إجابة 'كاملة' و 'مترابطة' وفق الشروط الصارمة التالية:
        1. العبارة الاستهلالية: ابدأ حصراً بـ: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        2. النقل والنسخ: انقل النصوص المرجعية 'حرفياً وبالكامل' دون أي تلخيص أو حذف للجمل الطويلة. استخدم روابط لغوية ذكية للربط بين هذه النصوص (مثل: كما يشير في موضع آخر، علاوة على ذلك يقرر...).
        3. التوثيق المتسلسل في المتن: ضع كل اقتباس حرفي بين علامتي تنصيص "" متبوعاً برقم مرجع متسلسل [1]، ثم [2] وهكذا.
           - ملاحظة: يجب أن يكون الترقيم متسلسلاً (1، 2، 3...) حسب ظهوره في الإجابة، ولا تكرر نفس الرقم أبداً؛ كل اقتباس جديد يأخذ رقماً جديداً حتى لو كان من نفس الصفحة.
        4. المراجع (الحاشية): في نهاية الإجابة، اكتب عنواناً بارزاً (المراجع:) ثم سرد المراجع المقابلة لكل رقم استخدمته بالصيغة: رقم المرجع- اسم المؤلف، اسم الكتاب، الجزء، ص: الصفحة.
        5. الصرامة: لا تضف أي معلومة خارجية أو تأويل. انقل القوائم المرقمة كما وردت تماماً.

        النصوص المرجعية المستخرجة من المكتبة:
        {ctx_text}

        سؤال الباحث:
        {user_query}
        """

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        for _ in range(3): 
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(15)
        
        return jsonify({"answer": "⚠️ الخادم مزدحم حالياً، يرجى المحاولة مرة أخرى."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
