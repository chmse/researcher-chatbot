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

# --- 2. تحميل المكتبة ---
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

# --- 3. محرك البحث الذكي (محسن لرفع الدقة) ---
def normalize(text):
    if not text: return ""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("[ةه]", "ه", text)
    text = re.sub("ى", "ي", text)
    return re.sub(r'[\u064B-\u0652]', '', text).strip()

def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    # تنظيف كلمات البحث للحصول على أدق النتائج
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي","بحث","دراسات"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        content = unit.get("content", "")
        content_norm = normalize(content)
        
        # حساب النقاط بناءً على تطابق الكلمات
        score = sum(5 for kw in keywords if kw in content_norm) 
        
        # دعم القوائم المرقمة
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', content.strip()): 
            score += 2
            
        if score > 0:
            # تحيز خفيف لترتيب الكتاب الأصلي
            score += (1 - (unit.get("page_pdf", 0) / 1000))
            scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    for _, idx in scored_indices[:top_k]:
        # جلب الوحدة المختارة + مسح تتبعي لـ 15 وحدة لضمان كمال الأفكار
        for i in range(max(0, idx-1), min(len(units), idx+15)):
            u_content = units[i].get("content", "")
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()):
                final_indices.add(i)
            # توقف ذكي إذا ابتعدنا عن الموضوع ولم نجد ترقيماً
            if i > idx + 5 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', u_content.strip()):
                if not any(kw in normalize(u_content) for kw in keywords):
                    break
                    
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال (المحرر الأكاديمي الصارم) ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        results = advanced_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        # بناء السياق المرجعي بدقة
        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += f"\n[مرجع رقم: {i+1}] [ص: {u.get('page_pdf','--')}] [كتاب: {u.get('book','--')}] [مؤلف: {u.get('author','--')}] [ج: {u.get('part','--')}]\n{u['content']}\n"
        
        # الموجه المعدل للصرامة القصوى وحفظ الهوية
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، قمت باستخراج المفاهيم والفقرات المرقمة من النصوص المرفقة بدقة متناهية، مع الالتزام بالقواعد المحددة:

        مهمتك الآن صياغة إجابة علمية كاملة بناءً على الأصول التالية:
        1. ابدأ الإجابة بالعبارة التالية نصاً: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، قمت باستخراج المفاهيم والفقرات المرقمة من النصوص المرفقة بدقة متناهية، رداً على سؤالكم:"
        2. المتن: انقل الجمل حرفياً كما وردت في المرجع دون تصرف، وضع كل جملة أو فقرة منقولة بين علامتي تنصيص مفردة ' ' متبوعة برقم المرجع الخاص بها [1].
        3. الشمولية: لا تختصر القوائم المرقمة. إذا وجدت (1، 2، 3، 4...) انقلها كاملة كما هي في النص المرفق.
        4. الحاشية: في نهاية الإجابة، اكتب عنوان (المراجع:) ثم سرد المراجع بالصيغة: رقم المرجع- اسم المؤلف، اسم الكتاب، الجزء، ص: رقم الصفحة.
        5. الصرامة: يمنع منعاً باتاً إضافة أي معلومة من خارج المرفقات أو تغيير المفردات اللسانية للمؤلف.

        النصوص المرجعية المستخرجة من المكتبة:
        {ctx_text}

        سؤال المستخدم:
        {user_query}
        """

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        for attempt in range(3):
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(15)
        
        return jsonify({"answer": "⚠️ ضغط كبير على الخادم، يرجى المحاولة بعد دقيقة."})

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
