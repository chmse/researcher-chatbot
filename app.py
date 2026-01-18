import os
import json
import re
import time
import requests # مكتبة إرسال الطلبات لـ Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# --- 1. إعدادات Groq API ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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
    print(f"📚 المكتبة جاهزة بـ {len(all_knowledge)} وحدة.")

load_kb()

def normalize(t): return re.sub("[إأآا]", "ا", re.sub("[ةه]", "ه", re.sub("ى", "ي", str(t or "")))).strip()

# --- 3. محرك البحث الاستكشافي (القوة المضافة) ---
def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    stop_words = {"ما","هي","أهم","مفهوم","في","على","من","إلى","عن","الذي","التي"}
    keywords = [w for w in query_norm.split() if w not in stop_words and len(w) > 2]
    
    scored_indices = []
    for idx, unit in enumerate(units):
        content = normalize(unit.get("content", ""))
        score = sum(5 for kw in keywords if kw in content)
        if re.match(r'^(\d+[-)]|[أ-ي][-)])', str(unit.get("content", "")).strip()): score += 2
        if score > 0: scored_indices.append((score, idx))
    
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = set()
    for _, idx in scored_indices[:top_k]:
        # سحب سياق موسع (2 قبل و 15 بعد) لضمان جلب كامل الفقرات والقوائم
        for i in range(max(0, idx-2), min(len(units), idx+15)):
            final_indices.add(i)
                
    return [units[i] for i in sorted(list(final_indices))]

# --- 4. دالة مناداة Groq (استخدام موديل Llama 3.3 70B القوي) ---
def ask_groq_model(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile", # واحد من أقوى النماذج المتوفرة مجاناً لديهم
        "messages": [
            {"role": "system", "content": "أنت محقق أكاديمي ملتزم بالنقل الحرفي الصارم وتوثيق المراجع."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0, # للحصول على نتائج دقيقة وليست خيالية
        "max_tokens": 4096
    }
    
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"❌ Groq Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

# --- 5. نقطة الاتصال الرئيسية ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        q = data.get("question")
        if not q: return jsonify({"answer": "لم يصل سؤال"}), 400

        results = advanced_search(q, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد ذلك في المكتبة."})

        ctx = ""
        for i, r in enumerate(results):
            ctx += f"\n[م:{i+1}] {r.get('author','--')} | {r.get('book','--')} | ج:{r.get('part','1')} | ص:{r.get('page_pdf','--')}\nالنص:{r.get('content','')}\n"

        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً رداً على سؤالكم:

        الأوامر:
        1. ابدأ حصراً بعبارة الترحيب الأكاديمية المطلوبة.
        2. انقل النصوص المرفقة حرفياً وبالكامل داخل "" متبوعة بالمرجع المتسلسل [1] وتجنب تكرار الأرقام.
        3. اربط بين الفقرات بلغة أكاديمية رصينة (روابط لغوية).
        4. في النهاية اذكر قائمة المراجع كاملة البيانات.
        
        النصوص المستخرجة: {ctx}
        سؤال الباحث: {q}"""

        # المحاولة عبر Groq
        answer = ask_groq_model(prompt)
        
        if answer:
            return jsonify({"answer": answer})
        return jsonify({"answer": "⚠️ محرك Groq لم يستجب حالياً، يرجى مراجعة مفتاح API."}), 500

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ فني: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
