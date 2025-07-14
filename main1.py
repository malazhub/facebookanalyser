
ACCESS_CODE = "12345"  # الكود الذي يجب على المستخدم إدخاله

from flask import Flask, render_template_string, request
from playwright.sync_api import sync_playwright, Page

import base64 
import pytesseract
#import requests
from io import BytesIO
import re
import threading
import webbrowser

import torch
import time
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util

#from PIL import ImageChops
# استبدال النموذج السابق بنموذج قوي للمطابقة المعنوية
transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

from PIL import Image
from datetime import datetime
import os
from rembg import remove

import imagehash






pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



   
def extract_text_with_boxes(image):
    if image is None:
        print("🚫 فشل تحميل الصورة (image is None)")
        return []

    try:
        # تحويل الصورة إلى RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # استخراج بيانات النص مع مواقع الصناديق
        data = pytesseract.image_to_data(image_rgb, lang='eng+ara', output_type=pytesseract.Output.DICT)

        all_texts = []
        grouped_blocks = []  # ← قائمة مضافة: لتجميع البلوكات النصية

        current_block_num = -1
        current_block_text = []

        # المرور على كل كلمة في البيانات
        for i in range(len(data['level'])):
            block_num = data['block_num'][i]
            text = data['text'][i].strip()

            # تخطي النص الفارغ
            if not text:
                continue

            if block_num != current_block_num:
                if current_block_text:
                    block_text = ' '.join(current_block_text)
                    all_texts.append(block_text)
                    grouped_blocks.append(current_block_text.copy())  # ← أضف البلوك كقائمة
                current_block_text = [text]
                current_block_num = block_num
            else:
                current_block_text.append(text)

        if current_block_text:
            block_text = ' '.join(current_block_text)
            all_texts.append(block_text)
            grouped_blocks.append(current_block_text.copy())  # ← أضف آخر بلوك

        print(f"📦 عدد الكتل النصية المكتشفة: {len(all_texts)}")

        # يمكنك هنا إعادة الاثنين معًا، أو الاحتفاظ بـ grouped_blocks للاستخدام الخارجي
        # return all_texts, grouped_blocks ← في حال أردت الإثنين معًا
        return all_texts, grouped_blocks
  # ← إذا أردت فقط all_texts كما هو ظاهر

    except Exception as e:
        print("❗ خطأ أثناء تحليل الصورة:", e)
        return []



    except Exception as e:
        print(f"❗ خطأ أثناء تحليل الصورة: {e}")
        return []



def extract_and_match_posts(all_texts, keywords, similarity_threshold):
    results = []

    # ✅ تحقق أولاً أن all_texts صالحة
    if not all_texts:
        print("⚠️ لا توجد نصوص لتحليلها (all_texts is empty or None).")
        return []

    

    # تأكد أن كل عنصر نصي
    try:
        texts = [t.strip() for t in all_texts if isinstance(t, str)]
    except Exception as e:
        print(f"❗ خطأ أثناء تجهيز النصوص: {e}")
        return []

    ##print("....texts....", texts)
    ##input(".......")

    for line in texts:
        if line.strip() == "":
            continue

        #
        #print(f"\n--- Individual Line ---\n{line}\n")

        for kw in keywords:
            sim = compute_similarity(kw, line)

            if sim >= similarity_threshold:
                results.append({
                    "matched_keyword": kw,
                    "similarity": sim,
                    "text": line
                })
                break  # ✅ توقف بعد أول تطابق ناجح

    return results




def split_into_blocks (text):
    return [block.strip() for block in re.split(r'\n\s*\n', text) if block.strip()]

def is_similar_ai(block, keywords, threshold=0.6):
    best_similarity = 0.0
    best_keyword = None
    for keyword in keywords:
        similarity = compute_similarity(block, keyword)
        if similarity > best_similarity:
            best_similarity = similarity
            best_keyword = keyword
    if best_similarity >= threshold:
        return True, best_similarity, best_keyword
    return False, 0.0, None

def extract_matching_blocks(text, keywords):
    blocks = split_into_blocks(text)
    matched = []
    for block in blocks:
        match, sim, keyword = is_similar_ai(block, keywords)
        if match:
            matched.append((block, sim, keyword))
    return matched


# إعداد Flask
app = Flask(__name__)







def screenshots_are_similar(img_bytes1, img_bytes2, threshold=0.98):
    # تحويل البايتات إلى صور PIL
    img1 = Image.open(BytesIO(img_bytes1)).convert("RGB")
    img2 = Image.open(BytesIO(img_bytes2)).convert("RGB")

    # حساب الـ perceptual hash لكلا الصورتين
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    # حساب الفرق بين الهاشات
    diff = hash1 - hash2
    hash_size = len(hash1.hash)**2  # عدد البتات في الهاش (عادة 64)
    
    similarity = 1 - (diff / hash_size)

    #print("🔍 pHash similarity:", similarity)
    #input("اضغط للمتابعة...")

    return similarity >= threshold



def apply_mask_to_image(image_bytes, mask):
    # تحويل الصورة من bytes إلى NumPy array بصيغة BGR (لـ OpenCV)
    image_stream = BytesIO(image_bytes)
    pil_image = Image.open(image_stream).convert("RGB")
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # التأكد أن أبعاد الصورة متوافقة مع القناع
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # إنشاء خلفية بيضاء
    background = np.ones_like(image, dtype=np.uint8) * 255

    # تحويل القناع إلى ثلاث قنوات
    if len(mask.shape) == 2:
        mask_3ch = cv2.merge([mask, mask, mask])
    else:
        mask_3ch = mask

    # تطبيق القناع
    foreground = cv2.bitwise_and(image, mask_3ch)
    background_masked = cv2.bitwise_and(background, cv2.bitwise_not(mask_3ch))
    final = cv2.add(foreground, background_masked)

    return final



def remove_background_with_rembg(image_bytes):
    # فتح الصورة من Bytes وتحويلها إلى RGB
    input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # إزالة الخلفية
    output_image = remove(input_image)

    # تحويل النتيجة إلى مصفوفة NumPy
    output_np = np.array(output_image)

    # استخراج قناة Alpha كقناع أبيض وأسود
    if output_np.shape[2] == 4:
        mask = output_np[:, :, 3]
    else:
        mask = np.ones(output_np.shape[:2], dtype=np.uint8) * 255

    return mask


def is_keyword_match(keyword, text):
    if keyword in text:
        return 1.0  # تطابق تام
    similarity = compute_similarity(keyword, text)
    return similarity


def compute_similarity(text1, text2, threshold=0.6):
    # تقسيم text2 إلى جمل أو مقاطع فرعية
    
    chunks = re.split(r'[.؟!\n]', text2)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # تضمين النصوص (text1 + كل أجزاء text2)
    texts = [text1] + chunks
    embeddings = transformer_model.encode(texts, convert_to_tensor=True)

    # حساب التشابه بين text1 وكل جزء من text2
    similarities = util.pytorch_cos_sim(embeddings[0], embeddings[1:])
    max_similarity = similarities.max().item()

    #print("...max similarity with parts of text2......", max_similarity)
    return max_similarity





def save_cookies(context):
    cookies = context.cookies()
    with open("cookies.json", "w", encoding="utf-8") as f:
        import json
        json.dump(cookies, f)

# بعد تسجيل الدخول يدويًا في متصفح غير headless:


def load_cookies(context):
    import json, os
    if os.path.exists("cookies.json"):
        with open("cookies.json", "r", encoding="utf-8") as f:
            cookies = json.load(f)
            context.add_cookies(cookies)


def expand_all_facebook_posts(page):
    try:
        elements = page.query_selector_all('div[role="button"]')
        for el in elements:
            try:
                text = el.inner_text().strip()
                if text in ["See more", "عرض المزيد"]:
                    el.scroll_into_view_if_needed()
                    page.wait_for_timeout(200)
                    el.click()
                    page.wait_for_timeout(300)
            except:
                continue
    except Exception as e:
        print("❌ خطأ:", e)



def analyze_posts_via_screenshot_ai(page: Page, keywords, max_posts, max_scrolls, similarity_threshold):

    seen_texts = set()
    scroll_count = 0
    analyzed_count = 0
    posts_data = []
    last_screenshot = None
    

    while analyzed_count < max_posts and scroll_count < max_scrolls:
        
        expand_all_facebook_posts(page)



        # ثم نأخذ screenshot كالمعتاد
        screenshot_bytes = page.screenshot(full_page=False)




      
        if last_screenshot and screenshots_are_similar(screenshot_bytes, last_screenshot):
            print("[Scroll Detection] Screenshot is similar. Skipping...")
            scroll_count += 1
            page.keyboard.press("PageDown")
            page.wait_for_timeout(1500)
            continue
        else:
            #print("[Scroll Detection] First screenshot.")
            last_screenshot = screenshot_bytes

        masked_image = None  # تأكد من وجود المتغير قبل try

        try:
            mask = remove_background_with_rembg(screenshot_bytes)
            masked_image = apply_mask_to_image(screenshot_bytes, mask)
        except Exception as e:
            print("Mask extraction failed:", e)
            try:
                image_stream = BytesIO(screenshot_bytes)
                pil_image = Image.open(image_stream).convert("RGB")
                masked_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e2:
                print("Fallback failed:", e2)
                masked_image = None

        
        if masked_image is not None:
            _, buffer = cv2.imencode('.png', masked_image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")
        else:
            image_base64 = ""


        all_texts, grouped_blocks = extract_text_with_boxes(masked_image)

        #print("...all text...",all_texts)
        #input(".....")

        for i, full_text in enumerate(all_texts):

            #print("...full text...",full_text)
            #input("......")
            for keyword in keywords:
                similarity = compute_similarity(full_text.strip(), keyword.strip())
                #print("...similarity....",similarity)
                #input("......")
                if similarity >= similarity_threshold:

                    
                    
                    matched_block = grouped_blocks[i]
                    #print("✅ Bloc مطابق:", ' '.join(matched_block))
                    #input(".....")

                   # ترميز الصورة إلى base64
                    image_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

                    posts_data.append({
                        "text": ' '.join(matched_block),
                        "matched_keyword": keyword,
                        "similarity": f"{similarity:.2f}",
                        "image_base64": image_base64,
                        "page_url": page.url
                    })

                    break


        analyzed_count += 1
        if analyzed_count < max_posts:
            page.keyboard.press("PageDown")
            page.wait_for_timeout(1500)

    posts_data.sort(key=lambda x: x.get("similarity", 0), reverse=True)


    return posts_data




# سكربت التشغيل الرئيسي بدون طباعة
def run_facebook_scraper(keywords, facebook_pages, max_posts, similarity_threshold):

    max_scrolls=5
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        
        load_cookies(context)
        page = context.new_page()
        
        all_results = []
       
        for fb_page in facebook_pages:
            page.goto(fb_page, timeout=60000)
            time.sleep(5)

           
            # ✅ يجب أن يكون هذا خارج الـ try
            posts_data = analyze_posts_via_screenshot_ai(page, keywords, max_posts, max_scrolls, similarity_threshold)


            all_results.extend(posts_data)

        browser.close()

        #cleanup_static_images()

        return all_results

def cleanup_static_images():
    folder = "static"
    for filename in os.listdir(folder):
        if filename.startswith("match_") and filename.endswith(".png"):
            os.remove(os.path.join(folder, filename))




# واجهة HTML بسيطة
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>

<script>
function openImage(src) {
  const w = window.open("", "_blank", "width=900,height=600,resizable=yes");
  w.document.write(`
    <html><head><title>Enlarged Image</title></head>
    <body style="margin:0;display:flex;justify-content:center;align-items:center;background:#000;">
      <img src="${src}" style="max-width:100%;max-height:100%;">
    </body></html>
  `);
}
</script>

    <meta charset="UTF-8">
    <title>Facebook Post Analyzer</title>
</head>
<body>
    <h2>Analyze Facebook Posts   /  malaz janbeih-malazjanbeih@gmail.com-+96170647081  </h2>
    <form method="POST">
        <label>Facebook Links (one per line):</label><br>
        <textarea name="links" rows="10" cols="60" style="border:1px solid #ccc; padding:5px; line-height:1.5; font-family: monospace; white-space: pre-wrap;"></textarea><br><br>
        <label>Keywords (separated by -):</label><br>
        <input type="text" name="keywords" size="60"><br><br>
        <label>Max Posts per Link:</label><br>
        <input type="number" name="max_posts" min="1" placeholder="Enter max posts"><br><br>
        <label>Similarity Threshold (default 0.7):</label><br>
        <input type="number" name="similarity" step="0.01" min="0.1" max="1.0" value="0.7"><br><br>
        <label>Enter Access Code:</label><br>
        <input type="text" name="access_code" required><br><br>

        <input type="submit" value="Analyze">
    </form>

   {% if message %}
    <p style="color:red;"><strong>{{ message }}</strong></p>
    {% endif %}

    {% for r in results %}
        <li>
            <strong>Matched Keyword:</strong> {{ r.matched_keyword }}<br>
            <strong>Text:</strong> <pre>{{ r.text }}</pre><br>
            <img src="data:image/png;base64,{{ r.image_base64 }}" width="300"
                style="cursor: zoom-in; border: 1px solid #ccc; border-radius: 5px;"
                onclick="openImage(this.src)">
        </li><hr>
    {% endfor %}

</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def home():
    message = ""
    results = []

    if request.method == "POST":
        access_code = request.form.get("access_code", "").strip()
        if access_code != ACCESS_CODE:
            message = "❌ Incorrect Access Code."
            return render_template_string(HTML_TEMPLATE, message=message)

        links = request.form["links"].strip().splitlines()
        keywords = request.form["keywords"].strip().split("-")
        max_posts = int(request.form["max_posts"])
        similarity_input = request.form.get("similarity", "0.7")
        try:
            similarity_threshold = float(similarity_input)
        except:
            similarity_threshold = 0.7

        results_holder = {}
        thread = threading.Thread(
            target=lambda: results_holder.update({
                "results": run_facebook_scraper(keywords, links, max_posts, similarity_threshold)
            })
        )
        thread.start()
        thread.join()
        results = results_holder.get("results", [])
        if not results:
            message = "❗ No matched keyword found."
        return render_template_string(HTML_TEMPLATE, results=results, message=message)

    return render_template_string(HTML_TEMPLATE)


def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=False)