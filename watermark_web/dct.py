import cv2
import numpy as np

STRENGTH = 150
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


# Block Processing Functions
def split_into_blocks(image, block_size=8):
    h, w = image.shape
    blocks = []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y + block_size, x:x + block_size]
            blocks.append(block.astype(np.float32))

    rows = h // block_size
    cols = w // block_size

    return blocks, rows, cols


def merge_blocks(blocks, rows, cols, block_size=8):
    height = rows * block_size
    width = cols * block_size

    image = np.zeros((height, width), dtype=np.float32)

    index = 0
    for y in range(rows):
        for x in range(cols):
            image[
                y * block_size:(y + 1) * block_size,
                x * block_size:(x + 1) * block_size
            ] = blocks[index]
            index += 1

    return np.clip(image, 0, 255).astype(np.uint8)


# DCT Functions
def apply_dct(block):
    return cv2.dct(block)


def apply_idct(dct_block):
    block = cv2.idct(dct_block)
    block = np.clip(block, 0, 255)

    return block.astype(np.float32)


def embed_bit_in_block(block, bit):
    dct_block = cv2.dct(block.copy())

    c1 = dct_block[2, 3]
    c2 = dct_block[3, 2]

    if bit == 1:
        if c1 <= c2 + STRENGTH:
            midpoint = (c1 + c2) / 2
            dct_block[2, 3] = midpoint + STRENGTH / 2
            dct_block[3, 2] = midpoint - STRENGTH / 2
    else:
        if c2 <= c1 + STRENGTH:
            midpoint = (c1 + c2) / 2
            dct_block[3, 2] = midpoint + STRENGTH / 2
            dct_block[2, 3] = midpoint - STRENGTH / 2

    return dct_block


def extract_bit_from_block(block):
    dct_block = cv2.dct(block.copy())

    c1 = dct_block[2, 3]
    c2 = dct_block[3, 2]

    return 1 if c1 > c2 else 0


def extract_raw_bits(image, max_bits=None):
    blocks, _, _ = split_into_blocks(image)

    extracted_bits = ""
    block_index = 0

    max_possible = len(blocks) // REDUNDANCY
    limit = max_possible if max_bits is None else min(max_bits, max_possible)

    for _ in range(limit):
        votes = []

        for _ in range(REDUNDANCY):
            bit = extract_bit_from_block(blocks[block_index])
            votes.append(bit)
            block_index += 1

        final_bit = 1 if sum(votes) > REDUNDANCY / 2 else 0
        extracted_bits += str(final_bit)

    return extracted_bits


# Main Watermarking Functions
def embed_message_dct(image, message):
    bits = build_message_packet(message)

    blocks, rows, cols = split_into_blocks(image)

    required_blocks = len(bits) * REDUNDANCY

    if required_blocks > len(blocks):
        raise ValueError("Message is too large for this image.")

    watermarked_blocks = []
    block_index = 0

    for bit_char in bits:
        bit = int(bit_char)

        for _ in range(REDUNDANCY):
            block = blocks[block_index]

            embedded_dct = embed_bit_in_block(block, bit)
            watermarked_block = apply_idct(embedded_dct)

            watermarked_blocks.append(watermarked_block)
            block_index += 1

    while block_index < len(blocks):
        watermarked_blocks.append(blocks[block_index])
        block_index += 1

    result = image.copy()
    result[:rows * 8, :cols * 8] = merge_blocks(watermarked_blocks, rows, cols)
    return result


def extract_message_dct(image):
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