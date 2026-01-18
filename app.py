import os
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات جوجل Gemini (النسخة الذكية لتجنب خطأ 404) ---
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

CANDIDATE_MODELS = [
    "models/gemini-1.5-flash",
    "gemini-1.5-flash",
    "models/gemini-pro",
    "gemini-pro"
]

def generate_with_fallback(prompt):
    """تجربة النماذج المتاحة حتى ينجح أحدهما"""
    for model_name in CANDIDATE_MODELS:
        try:
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(prompt, generation_config={"temperature": 0.0})
            if response: return response.text
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e).lower():
                continue
            else:
                time.sleep(2)
    return None

# --- 2. تحميل المكتبة الكاملة ---
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
    print(f"📚 تم تحميل {len(all_knowledge)} وحدة معرفية.")

load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

# --- 3. محرك البحث الاستكشافي (القوة المضافة) ---
def advanced_search(query, units, top_k=3):
    """يبحث عن الكلمات المفتاحية ويسحب القوائم والفقرات المرتبطة (15 فقرة بعد)"""
    query_norm = normalize(query)
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        content = normalize(unit.get("content", ""))
        score = sum(5 for kw in keywords if kw in content)
        # إعطاء أفضلية للفقرات المرقبة (أرقام أو حروف)
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', str(unit.get("content", "")).strip()): score += 2
        if score > 0: scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    for _, idx in scored_indices[:top_k]:
        # التوسع: سحب فقرتين قبل (للسياق) و15 فقرة بعد (لضمان جلب كامل القائمة)
        for i in range(max(0, idx-2), min(len(units), idx+15)):
            u_content = units[i].get("content", "")
            # ضم الفقرة إذا كانت مرتبطة بكلمات البحث أو بتسلسل ترقيمي
            if i == idx or re.match(r'^(\d+[-)]|[أ-ي][-)])', str(u_content).strip()) or any(k in normalize(u_content) for k in keywords):
                final_indices.add(i)
            # التوقف إذا ابتعدنا كثيراً وانقطع الترقيم
            if i > idx + 8 and not re.match(r'^(\d+[-)]|[أ-ي][-)])', str(u_content).strip()): break
                
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. معالجة الطلب وصياغة الإجابة ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        q = data.get("question")
        if not q: return jsonify({"answer": "لم يصل سؤال"}), 400

        # استخدام البحث المطور
        results = advanced_search(q, all_knowledge)
        if not results: return jsonify({"answer": "لم أجد هذه المعلومة في المكتبة."})

        # بناء مرجع النصوص
        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n[م:{i+1}] {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\nنص:{r.get('content','')}\n"

        # موجه الأوامر (الصارم)
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً استجابةً لسؤالكم:

        الشروط والأوامر:
        1. ابدأ حصراً بعبارة الترحيب الأكاديمية: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        2. انسخ النصوص المرفقة بالأسفل "حرفياً وبالكامل" كما هي دون تلخيص أو تغيير للألفاظ.
        3. اربط بين الاقتباسات بذكاء لغوي (وفي هذا الصدد، كما يوضح في موضع آخر، علاوة على...).
        4. ضع النص المنقول بين "" متبوعاً برقم متسلسل [1]، [2] وهكذا (لا تكرر الأرقام، كل اقتباس له رقم جديد).
        5. في النهاية اذكر المراجع كاملة ببياناتها (المؤلف، الكتاب، ص، ج).

        المادة المرجعية المرفقة:
        {ctx}

        سؤال الباحث: {q}
        """

        answer = generate_with_fallback(prompt)
        
        if answer: return jsonify({"answer": answer})
        return jsonify({"answer": "⚠️ فشل المحرك في توليد الإجابة."}), 500

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ فني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
