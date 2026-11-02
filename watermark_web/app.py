from flask import Flask, render_template, request, session
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from attack_utils import (
    jpeg_compression_attack,
    blurring_attack,
    salt_pepper_attack,
    rotation_attack,
    translation_attack,
    PSNR,
    embed_watermark,
    extract_watermark,
    text_to_bits
)
from dct import (
    load_grayscale_image,
    embed_message_dct,
    extract_message_dct,
    save_image
)

from dwt import embed_message_dwt, extract_message_dwt

TRANSFORMS = {"dct": (embed_message_dct, extract_message_dct),
              "dwt": (embed_message_dwt, extract_message_dwt)}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "watermark_secret_key"

# ================= STATIC DIR =================
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# ================= GLOBAL =================
original_watermark_global = ""

# ================= METRICS =================
def calculate_psnr(original, processed):
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(original, processed):
    gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score


def BER(original_bits, extracted_bits):
    original_bits = np.array(list(original_bits))
    extracted_bits = np.array(list(extracted_bits))

    min_len = min(len(original_bits), len(extracted_bits))
    total = max(len(original_bits), len(extracted_bits))
    errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len]) + abs(len(original_bits) - len(extracted_bits))

    return errors / total if total else 0


def normalized_correlation(w, w_hat):
    w = np.array(list(w), dtype=float)
    w_hat = np.array(list(w_hat), dtype=float)

    min_len = min(len(w), len(w_hat))
    w = w[:min_len]
    w_hat = w_hat[:min_len]

    if min_len == 0:
        return 0

    num = np.sum((w - np.mean(w)) * (w_hat - np.mean(w_hat)))
    den = np.sqrt(np.sum((w - np.mean(w))**2) * np.sum((w_hat - np.mean(w_hat))**2))

    return num / den if den != 0 else 0


@app.errorhandler(ValueError)
def invalid_input(error):
    return str(error), 400


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= EMBED =================
# ================= EMBED =================
@app.route("/embed", methods=["GET", "POST"])
def embed():
    if request.method == "POST":

        file = request.files.get("image")
        if not file or not file.filename:
            raise ValueError("Please upload an image.")
        watermark_text = request.form["watermark"]
        method = request.form.get("method", "lsb")

        if method not in {"lsb", "dct", "dwt"}:
            raise ValueError("Unknown watermark method.")
        os.makedirs(STATIC_DIR, exist_ok=True)

        original_path = os.path.join(STATIC_DIR, "original.png")
        embedded_path = os.path.join(STATIC_DIR, "embedded.png")

        file.save(original_path)

        # embedding
        if method in TRANSFORMS:
            image = load_grayscale_image(original_path)
            watermarked = TRANSFORMS[method][0](image, watermark_text)
            save_image(watermarked, embedded_path)
        else:
            embed_watermark(original_path, embedded_path, watermark_text)

        # session
        session["watermark"] = watermark_text
        session["embedded_path"] = embedded_path
        session["method"] = method

        # metrics
        if method in TRANSFORMS:
            original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            embedded = cv2.imread(embedded_path, cv2.IMREAD_GRAYSCALE)
            mse = np.mean((original.astype(np.float64) - embedded.astype(np.float64)) ** 2)
            psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
            ssim_val, _ = ssim(original, embedded, full=True)
        else:
            original = cv2.imread(original_path)
            embedded = cv2.imread(embedded_path)
            psnr = calculate_psnr(original, embedded)
            ssim_val = calculate_ssim(original, embedded)

        # extraction
        if method in TRANSFORMS:
            image = load_grayscale_image(embedded_path)
            extracted_text = TRANSFORMS[method][1](image)
        else:
            extracted_text = extract_watermark(embedded_path)

        original_bits = text_to_bits(watermark_text)
        extracted_bits = text_to_bits(extracted_text)

        ber = BER(original_bits, extracted_bits)
        nc = normalized_correlation(original_bits, extracted_bits)

        return render_template(
            "result.html",
            psnr=round(psnr, 2),
            ssim=round(ssim_val, 4),
            ber=round(ber, 4),
            nc=round(nc, 4)
        )

    return render_template("embed.html")


# ================= ATTACK =================
@app.route("/attack", methods=["GET", "POST"])
def attack():

    if request.method == "POST":

        embedded_path = session.get("embedded_path")

        if not embedded_path:
            return "Error: Please embed first"

        method = session.get("method", "lsb")

        if method in TRANSFORMS:
            embedded = cv2.imread(embedded_path, cv2.IMREAD_GRAYSCALE)
        else:
            embedded = cv2.imread(embedded_path)
        attacked_path = os.path.join(STATIC_DIR, "attacked.png")

        attack_type = request.form.get("attack_type", "blur")

        if attack_type == "jpeg":
            attacked = jpeg_compression_attack(embedded)

        elif attack_type == "noise":
            output_path = os.path.join(STATIC_DIR, "attacked.png")
            salt_pepper_attack(embedded_path, output_path)
            attacked = cv2.imread(output_path)

        elif attack_type == "rotation":
            attacked = rotation_attack(embedded)

        elif attack_type == "translation":
            attacked = translation_attack(embedded)

        else:
            attacked = blurring_attack(embedded)

        cv2.imwrite(attacked_path, attacked)

        if method in TRANSFORMS:
            attacked = cv2.imread(attacked_path, cv2.IMREAD_GRAYSCALE)

# ================= QUALITY =================
        if method in TRANSFORMS:
           mse = np.mean((embedded.astype(np.float64) - attacked.astype(np.float64)) ** 2)
           psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
           ssim_val, _ = ssim(embedded, attacked, full=True)
        else:
           psnr = calculate_psnr(embedded, attacked)
           ssim_val = calculate_ssim(embedded, attacked)

        # ================= WATERMARK ROBUSTNESS =================
        method = session.get("method", "lsb")

        if method in TRANSFORMS:
            image = load_grayscale_image(attacked_path)
            extracted_text = TRANSFORMS[method][1](image)
        else:
            extracted_text = extract_watermark(attacked_path)

        original_watermark = session.get("watermark", "")

        original_bits = text_to_bits(original_watermark)
        extracted_bits = text_to_bits(extracted_text)

        ber = BER(original_bits, extracted_bits)
        nc = normalized_correlation(original_bits, extracted_bits)

        return render_template(
            "attack_result.html",
            embedded="embedded.png",
            attacked="attacked.png",
            psnr=round(psnr, 2),
            ssim=round(ssim_val, 4),
            ber=round(ber, 4),
            nc=round(nc, 4)
        )

    return render_template("attack.html")


# ================= EXTRACT =================
@app.route("/extract", methods=["GET", "POST"])
def extract():

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            return "Error: Please upload an image."

        extract_path = os.path.join(STATIC_DIR, "to_extract.png")
        file.save(extract_path)

        method = request.form.get("method") or session.get("method", "lsb")

        if method in TRANSFORMS:
            image = load_grayscale_image(extract_path)
            watermark = TRANSFORMS[method][1](image)
        else:
            watermark = extract_watermark(extract_path)

        return render_template(
            "extract_result.html",
            watermark=watermark
        )

    return render_template("extract.html")

if __name__ == "__main__":
    app.run(debug=True)