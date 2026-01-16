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

# --- 1. إعدادات جوجل Gemini مع إصلاح خطأ الـ 404 ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_model_name():
    """دالة ذكية لاكتشاف اسم النموذج الصحيح وتجنب خطأ 404"""
    try:
        # محاولة سرد النماذج المتاحة لمفتاحك
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # البحث عن نموذج flash 1.5
        for m in available_models:
            if 'gemini-1.5-flash' in m:
                return m
        
        # إذا لم يجد، نستخدم الاسم المختصر كبديل
        return "gemini-1.5-flash"
    except Exception as e:
        print(f"⚠️ فشل اكتشاف النماذج: {e}")
        return "gemini-1.5-flash"

MODEL_NAME = get_model_name()
print(f"✅ تم اختيار النموذج: {MODEL_NAME}")

# --- 2. تحميل المكتبة الكاملة ---
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

# --- 3. محرك البحث الذكي ---
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
        if not data: return jsonify({"answer": "خطأ في البيانات"}), 400
        
        user_query = data.get("question")
        if not user_query: return jsonify({"answer": "لم يصل سؤال"}), 400

        results = advanced_search(user_query, all_knowledge)
        if not results: return jsonify({"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."})

        ctx_text = ""
        for i, u in enumerate(results):
            author = u.get('author', '--')
            book = u.get('book', '--')
            part = u.get('part', '1')
            page = u.get('page_pdf', '--')
            content = u.get('content', '')
            ctx_text += f"\n--- [مرجع: {i+1}] ---\nالمؤلف: {author} | الكتاب: {book} | ج: {part} | ص: {page}\nالنص: {content}\n"
        
        # الموجه (Prompt) المدمج: يجمع بين الربط اللغوي والنسخ الحرفي الكامل
        prompt = f"""بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية، إليكم عرضاً موثقاً استجابةً لسؤالكم:

        مهمتك صياغة إجابة 'كاملة' و 'مترابطة' وفق الشروط الصارمة التالية:
        1. العبارة الاستهلالية: ابدأ حصراً بـ: "بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح، واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة، إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:"
        2. الربط والنسخ: استخدم أدوات الربط اللغوية (مثل: وفي هذا السياق، علاوة على ذلك، كما يقرر في موضع آخر...) لربط الأفكار، ولكن عندما تنقل المعلومة من النص المرفق، انقلها 'حرفياً' وبالكامل دون أي اختصار أو تلخيص.
        3. التوثيق المتسلسل (المتن): ضع كل نص منقول حرفياً بين علامتي تنصيص "" متبوعاً برقم مرجع متسلسل [1]، ثم [2]، وهكذا. 
           - هام: لا تكرر الرقم أبداً. كل اقتباس جديد يأخذ رقماً جديداً (1، 2، 3...) حتى لو كان من نفس الصفحة.
        4. عدم الضياع: انقل القوائم والتعليلات كما وردت في النصوص المرفقة كاملةً.
        5. الحاشية (المراجع): في نهاية الإجابة، اكتب عنواناً بارزاً (المراجع:) ثم اذكر المراجع المقابلة للأرقام بالصيغة: رقم المرجع- اسم المؤلف، اسم الكتاب، الجزء، ص: رقم الصفحة.
        6. الصرامة: ممنوع إضافة أي معلومة من خارج النصوص المرفقة أو تأويل شخصي.

        سؤال الباحث: {user_query}
        """

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        for _ in range(3): 
            try:
                response = model.generate_content(prompt, generation_config={"temperature": 0.0})
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(10)
            except Exception as e:
                # محاولة أخيرة باسم نموذج مختلف إذا فشل الأول
                if "not found" in str(e).lower():
                    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    return jsonify({"answer": response.text})
                raise e
        
        return jsonify({"answer": "⚠️ الخادم مزدحم، حاول مجدداً."})

    except Exception as e:
        print(f"❌ Error during request: {e}")
        return jsonify({"answer": f"❌ حدث خطأ: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
