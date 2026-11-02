import cv2
import numpy as np
from PIL import Image
from dct import (
    load_grayscale_image,
    embed_message_dct,
    extract_message_dct,
    save_image
)

# ================= PSNR =================
def PSNR(original, compressed):
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# ================= JPEG ATTACK =================
def jpeg_compression_attack(image, quality=10):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encimg = cv2.imencode('.jpg', image, encode_param)

    if not success:
        raise ValueError("JPEG compression failed")

    return cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR)


# ================= BLUR ATTACK =================
def blurring_attack(image, kernel_size=3):
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# ================= SALT & PEPPER NOISE =================
def add_noise(image_path, output_path, intensity=50):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    noise = np.random.randint(-intensity, intensity, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    noisy = Image.fromarray(arr)
    noisy.save(output_path)
    print("✔️ Noise added")

	
	
def salt_pepper_attack(image_path, output_path, amount=0.08):
    img  = Image.open(image_path).convert("RGB")
    arr  = np.array(img)                       
    h, w = arr.shape[:2]
    num  = int(amount * h * w)

    # Salt — set random pixels to 255
    coords = [np.random.randint(0, h, num // 2),
              np.random.randint(0, w, num // 2)]
    arr[coords[0], coords[1]] = 255

    # Pepper — set random pixels to 0
    coords = [np.random.randint(0, h, num // 2),
              np.random.randint(0, w, num // 2)]
    arr[coords[0], coords[1]] = 0

    Image.fromarray(arr).save(output_path)     
    print("✔ Salt & pepper attack done")	


# ================= ROTATION =================
def rotation_attack(image, angle=90):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


# ================= TRANSLATION =================
def translation_attack(image, tx=8, ty=8):
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    h, w = image.shape[:2]

    return cv2.warpAffine(
        image,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )


# ================= WATERMARK =================

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)


def bits_to_text(bits):
    text = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text


def embed_watermark(image_path, output_path, text):
    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())

    bits = text_to_bits(text)
    payload = format(len(bits), '032b') + bits

    new_pixels = []
    i = 0

    for (r, g, b) in pixels:
        if i < len(payload):
            r = (r & ~1) | int(payload[i]); i += 1
        if i < len(payload):
            g = (g & ~1) | int(payload[i]); i += 1
        if i < len(payload):
            b = (b & ~1) | int(payload[i]); i += 1

        new_pixels.append((r, g, b))

    out = Image.new("RGB", img.size)
    out.putdata(new_pixels)
    out.save(output_path)

    return output_path


def extract_watermark(image_path):
    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())

    bits = ""
    for (r, g, b) in pixels:
        bits += str(r & 1)
        bits += str(g & 1)
        bits += str(b & 1)

    length = int(bits[:32], 2)
    message_bits = bits[32:32 + length]

    return bits_to_text(message_bits)
    
    
def image_to_bits(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
    return binary.flatten()
    