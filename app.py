import os
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.api_core import exceptions

# ===============================
# 1) تهيئة التطبيق
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# 2) إعداد Gemini
# ===============================
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GEMINI_API_KEY غير موجود في متغيرات البيئة")

genai.configure(api_key=GOOGLE_API_KEY)


def get_model_name():
    try:
        models = [
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for m in models:
            if "1.5-flash" in m:
                return m
        return "models/gemini-1.5-flash"
    except Exception:
        return "models/gemini-1.5-flash"


MODEL_NAME = get_model_name()

# ===============================
# 3) تحميل المكتبة (عند الإقلاع فقط – متوافق مع Flask 3)
# ===============================
all_knowledge = []
KB_PATH = "library_knowledge"


def load_library():
    global all_knowledge
    all_knowledge = []
    if os.path.exists(KB_PATH):
        for filename in sorted(os.listdir(KB_PATH)):
            if filename.endswith(".json"):
                with open(
                    os.path.join(KB_PATH, filename),
                    "r",
                    encoding="utf-8",
                ) as f:
                    all_knowledge.extend(json.load(f))


load_library()

# ===============================
# 4) أدوات المعالجة النصية
# ===============================
def normalize(text: str) -> str:
    if not text:
        return ""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("[ةه]", "ه", text)
    text = re.sub("ى", "ي", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)  # حذف التشكيل
    return text.strip()


def advanced_search(query, units, top_k=3):
    query_norm = normalize(query)
    stop_words = {
        "ما",
        "هي",
        "أهم",
        "مفهوم",
        "في",
        "على",
        "من",
        "إلى",
        "عن",
        "الذي",
        "التي",
    }
    keywords = [
        w for w in query_norm.split() if w not in stop_words and len(w) > 2
    ]

    scored_indices = []
    for idx, unit in enumerate(units):
        content = unit.get("content", "")
        content_norm = normalize(content)
        score = sum(5 for kw in keywords if kw in content_norm)
        if re.match(r"^(\d+[-)]|[أ-ي][-)])", content.strip()):
            score += 2
        if score > 0:
            scored_indices.append((score, idx))

    scored_indices.sort(key=lambda x: x[0], reverse=True)

    final_indices = set()
    for _, idx in scored_indices[:top_k]:
        for i in range(max(0, idx - 2), min(len(units), idx + 15)):
            u_content = units[i].get("content", "")
            if (
                i == idx
                or re.match(r"^(\d+[-)]|[أ-ي][-)])", u_content.strip())
                or any(k in normalize(u_content) for k in keywords)
            ):
                final_indices.add(i)
            if i > idx + 7 and not re.match(
                r"^(\d+[-)]|[أ-ي][-)])", u_content.strip()
            ):
                break

    return [units[i] for i in sorted(final_indices)]


# ===============================
# 5) نقطة الاتصال
# ===============================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_query = data.get("question")
        if not user_query:
            return jsonify({"answer": "لم يصل سؤال."}), 400

        results = advanced_search(user_query, all_knowledge)
        if not results:
            return jsonify(
                {"answer": "عذراً، لم أجد هذه المعلومة في المكتبة."}
            )

        ctx_text = ""
        for i, u in enumerate(results):
            ctx_text += (
                f"\n--- [معرف المرجع: {i+1}] ---\n"
                f"المؤلف: {u.get('author','--')} | "
                f"الكتاب: {u.get('book','--')} | "
                f"ج: {u.get('part','--')} | "
                f"ص: {u.get('page_pdf','--')}\n"
                f"النص: {u.get('content','')}\n"
            )

        prompt = f"""
بصفتي باحثاً أكاديمياً في فكر الأستاذ الدكتور عبد الرحمن الحاج صالح،
واستناداً إلى المنهجية اللسانية الاستقرائية في تحليل المتون المرفقة،
إليكم عرضاً موثقاً للأصول العلمية رداً على سؤالكم:

الشروط الصارمة:
1) انقل النصوص المرجعية حرفياً وبالكامل دون تلخيص.
2) ضع كل اقتباس بين علامتي تنصيص "" مع ترقيم متسلسل [1]، [2]...
3) الربط اللغوي مسموح فقط خارج علامات التنصيص.
4) لا تضف أي معلومة خارج النصوص المرفقة.
5) في النهاية أدرج (المراجع:) مع مطابقة الأرقام.

النصوص المرجعية:
{ctx_text}

سؤال الباحث:
{user_query}
"""

        model = genai.GenerativeModel(model_name=MODEL_NAME)

        for _ in range(3):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.0},
                )
                return jsonify({"answer": response.text})
            except exceptions.TooManyRequests:
                time.sleep(15)

        return jsonify(
            {"answer": "⚠️ الخادم مزدحم حالياً، يرجى المحاولة لاحقاً."}
        )

    except Exception as e:
        return jsonify({"answer": f"❌ خطأ تقني: {str(e)}"}), 500


# ===============================
# 6) التشغيل
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
