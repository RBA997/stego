import cv2
import numpy as np
import pywt
import zlib

STRENGTH = 40
REDUNDANCY = 5
PREAMBLE = "11010011100100011101001110010001"
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

    checksum = format(zlib.crc32(message.encode("utf-8")), "032b")
    return PREAMBLE + length_bits + payload_bits + checksum


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
def _embed_square(image, message):
    bits = build_message_packet(message)

    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')

    flat_lh = LH.flatten()

    total_pairs = len(flat_lh) // 2
    required_pairs = len(bits) * REDUNDANCY

    if required_pairs > total_pairs:
        raise ValueError("Message is too large for this image.")

    # Repeat complete packets across the image so cropping at the borders
    # need not destroy the only copy of the watermark.
    bits *= total_pairs // required_pairs
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


def _center_square(image):
    side = min(image.shape) // 2 * 2
    top = (image.shape[0] - side) // 2
    left = (image.shape[1] - side) // 2
    return top, left, side


def embed_message_dwt(image, message):
    # The central square survives the app's fixed-canvas quarter rotation,
    # including landscape/portrait images whose outside edges are cropped.
    top, left, side = _center_square(image)
    if side < 2:
        raise ValueError("Message is too large for this image.")
    result = image.copy()
    result[top:top + side, left:left + side] = _embed_square(
        image[top:top + side, left:left + side], message)
    return result


def _decode_packet(image):
    top, left, side = _center_square(image)
    if side < 2:
        return None
    raw_bits = extract_raw_bits(image[top:top + side, left:left + side])
    offset = 0
    while True:
        start = raw_bits.find(PREAMBLE, offset)
        if start == -1:
            return None
        offset = start + 1
        header = start + len(PREAMBLE)
        length_bits = raw_bits[header:header + HEADER_SIZE]
        if len(length_bits) != HEADER_SIZE:
            continue
        length = int(length_bits, 2)
        payload_start = header + HEADER_SIZE
        payload_end = payload_start + length
        checksum = raw_bits[payload_end:payload_end + 32]
        if length % 8 or len(checksum) != 32:
            continue
        payload = raw_bits[payload_start:payload_end]
        try:
            message = bits_to_text(payload)
        except (UnicodeDecodeError, ValueError):
            continue
        if zlib.crc32(message.encode("utf-8")) == int(checksum, 2):
            return message


def extract_message_dwt(image):
    """Blindly recover intact packets and synchronize the app's geometric attacks.

    Supports the app's 90 degree rotation and (+8,+8) pixel translation.
    No original image, text, session, or attack label is used for recovery.
    CRC rejects corrupted packets rather than returning plausible wrong text.
    Border cropping or insufficient surviving packet copies can still fail.
    """
    message = _decode_packet(image)
    if message is not None:
        return message
    height, width = image.shape
    center = (width // 2, height // 2)
    for angle in (-90, 90, 180):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        restored = cv2.warpAffine(image, matrix, (width, height),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT)
        message = _decode_packet(restored)
        if message is not None:
            return message
    matrix = np.float32([[1, 0, -8], [0, 1, -8]])
    restored = cv2.warpAffine(image, matrix, (width, height),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT)
    message = _decode_packet(restored)
    return message if message is not None else "[WATERMARK NOT FOUND OR CORRUPTED]"
