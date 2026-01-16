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

# --- 1. إعدادات جوجل Gemini ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_model_name():
    try:
        # محاولة اختيار أفضل نموذج متاح
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

MODEL_NAME = get_model_name()

# --- 2. تحميل المكتبة الكاملة مع الحماية ---
all_knowledge = []
KB_PATH = "library_knowledge"

def load_library():
    global all_knowledge
    all_knowledge = []
    if os.path.exists(KB_PATH):
        for filename in sorted(os.listdir(KB_PATH)):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(KB_PATH, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_knowledge.extend(data)
                except Exception as e:
                    print(f"⚠️ خطأ في تحميل ملف {filename}: {e}")
    return len(all_knowledge)

load_library()

# --- 3. محرك البحث الذكي (محمي من الحقول المفقودة) ---
def normalize(text):
    if not text: return ""
    text = str(text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("[ةه]", "ه", text)
    text = re.sub("ى", "ي", text)
    return re.sub(r'[\u064B-\u0652]', '', text).strip()

def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        # استخدام .get لضمان عدم الانهيار إذا فقد الحقل
        content = unit.get("content", "")
        if not content: continue
        
        content_norm = normalize(content)
        score = sum(5 for kw in keywords if kw in content_norm)
        
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', str(content).strip()): score += 2
        if score > 0:
            scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    for _, idx in scored_indices[:top_k]:
        for i in range(max(0, idx-2), min(len(units), idx+15)):
            u_content = units[i].get("content", "")
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', str(u_content).strip()) or any(k in normalize(u_content) for k in keywords):
                final_indices.add(i)
            if i > idx + 7 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', str(u_content).strip()):
                break
                
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. نقطة الاتصال ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        if not data: return jsonify({"answer": "خطأ في استقبال البيانات."}), 400
        
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال."}), 400

        if not all_knowledge:
            return jsonify({"answer": "المكتبة فارغة، تأكد من رفع ملفات الـ JSON في المجلد الصحيح."})

        results = advanced_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة المرفوعة."})

        # بناء السياق مع حماية كاملة لكل الحقول
        ctx_text = ""
        for i, u in enumerate(results):
            author = u.get('author', 'غير متوفر')
            book = u.get('book', 'غير متوفر')
            part = u.get('part', '1')
            page = u.get('page_pdf', '--')
            content = u.get('content', '[نص مفقود]')
            
            ctx_text += f"\n--- [معرف المرجع: {i+1}] ---\nالمؤلف: {author} | الكتاب: {book} | ج: {part} | ص: {page}\nالنص: {content}\n"
        
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية، إليكم عرضاً موثقاً استجابةً لسؤالكم:

        التعليمات:
        1. ابدأ بـ: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        2. استخدم الربط اللغوي ولكن انقل النصوص المرجعية 'حرفياً' وبالكامل.
        3. ضع كل اقتباس حرفي بين "" متبوعاً برقم مرجع [1]، [2]... دون تكرار الرقم.
        4. في النهاية اكتب (المراجع:) وسردها بالصيغة: رقم المرجع- المؤلف، الكتاب، الجزء، ص: الصفحة.

        المادة العلمية:
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
                time.sleep(10)
        
        return jsonify({"answer": "⚠️ الخادم مزدحم، يرجى إعادة المحاولة."})

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"answer": f"❌ حدث خطأ داخلي: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
