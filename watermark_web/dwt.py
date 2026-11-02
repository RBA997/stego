import cv2
import numpy as np
import pywt

STRENGTH = 40
REDUNDANCY = 5
PREAMBLE = "1010101010101010"
HEADER_SIZE = 16


# Image Processing Functions
def load_grayscale_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image.astype(np.uint8)


def pad_to_multiple(image, multiple=8):
    height, width = image.shape

    pad_h = (multiple - (height % multiple)) % multiple
    pad_w = (multiple - (width % multiple)) % multiple

    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT
    )

    return padded


def save_image(image, path):
    cv2.imwrite(path, image)


# Text Conversion Functions
def text_to_bits(text):
    return ''.join(format(byte, '08b') for byte in text.encode('utf-8'))


def bits_to_text(bits):
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)).decode('utf-8')


def int_to_16bit_binary(number):
    return format(number, "016b")


def build_message_packet(message):
    payload_bits = text_to_bits(message)
    if len(payload_bits) > 65535:
        raise ValueError("Message exceeds the 16-bit packet length limit.")
    length_bits = int_to_16bit_binary(len(payload_bits))

    return PREAMBLE + length_bits + payload_bits


# DWT Functions
def embed_bit_in_pair(c1, c2, bit):
    if bit == 1:
        if c1 <= c2 + STRENGTH:
            midpoint = (c1 + c2) / 2
            c1 = midpoint + STRENGTH / 2
            c2 = midpoint - STRENGTH / 2
    else:
        if c2 <= c1 + STRENGTH:
            midpoint = (c1 + c2) / 2
            c2 = midpoint + STRENGTH / 2
            c1 = midpoint - STRENGTH / 2

    return c1, c2


def extract_bit_from_pair(c1, c2):
    return 1 if c1 > c2 else 0


# Main Watermarking Functions
def embed_message_dwt(image, message):
    bits = build_message_packet(message)

    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')

    flat_lh = LH.flatten()

    total_pairs = len(flat_lh) // 2
    required_pairs = len(bits) * REDUNDANCY

    if required_pairs > total_pairs:
        raise ValueError("Message is too large for this image.")

    pair_index = 0

    for bit_char in bits:
        bit = int(bit_char)

        for _ in range(REDUNDANCY):
            idx1 = pair_index * 2
            idx2 = idx1 + 1

            c1 = flat_lh[idx1]
            c2 = flat_lh[idx2]

            new_c1, new_c2 = embed_bit_in_pair(c1, c2, bit)

            flat_lh[idx1] = new_c1
            flat_lh[idx2] = new_c2

            pair_index += 1

    LH = flat_lh.reshape(LH.shape)

    watermarked = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)

    return watermarked[:image.shape[0], :image.shape[1]]


def extract_raw_bits(image, max_bits=None):
    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')

    flat_lh = LH.flatten()

    total_pairs = len(flat_lh) // 2
    max_possible = total_pairs // REDUNDANCY
    limit = max_possible if max_bits is None else min(max_bits, max_possible)

    extracted_bits = ""
    pair_index = 0

    for _ in range(limit):
        votes = []

        for _ in range(REDUNDANCY):
            idx1 = pair_index * 2
            idx2 = idx1 + 1

            c1 = flat_lh[idx1]
            c2 = flat_lh[idx2]

            bit = extract_bit_from_pair(c1, c2)
            votes.append(bit)

            pair_index += 1

        final_bit = 1 if sum(votes) > REDUNDANCY / 2 else 0
        extracted_bits += str(final_bit)

    return extracted_bits


def extract_message_dwt(image):
    raw_bits = extract_raw_bits(image)

    start = raw_bits.find(PREAMBLE)

    if start == -1:
        return "[PREAMBLE NOT FOUND]"

    start += len(PREAMBLE)

    length_bits = raw_bits[start:start + HEADER_SIZE]

    if len(length_bits) < HEADER_SIZE:
        return "[HEADER CORRUPTED]"

    payload_length = int(length_bits, 2)

    start += HEADER_SIZE

    payload_bits = raw_bits[start:start + payload_length]

    if len(payload_bits) < payload_length:
        return "[PAYLOAD CORRUPTED]"

    if payload_length % 8:
        return "[PAYLOAD CORRUPTED]"
    try:
        return bits_to_text(payload_bits)
    except UnicodeDecodeError:
        return "[PAYLOAD CORRUPTED]"