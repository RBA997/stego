# Watermark web app

This is the editable application extracted from the repository's original ZIP. The original templates and styles are preserved.

## Run

From `/workspace/stego`:

```sh
python -m venv /workspace/stego-venv
/workspace/stego-venv/bin/pip install -r watermark_web/requirements.txt
cd watermark_web
/workspace/stego-venv/bin/python -m flask --app app run --host 0.0.0.0 --port 5000
```

Run tests from the repository root:

```sh
/workspace/stego-venv/bin/python -m pytest watermark_web/tests -q
```

Choose DCT or DWT on Embed, then upload the downloaded PNG on Extract in the same browser session. API clients may also send `method=dct` or `method=dwt` to `/extract`. DCT uses complete 8×8 blocks and preserves remaining edge pixels. DWT uses Haar detail-coefficient pairs in the central even-sized square, repeating complete packets where capacity permits. Its packet includes a CRC32 checksum. DWT capacity is therefore based on this square, rather than the full rectangular image. Both encode UTF-8 text with a 16-bit payload length and five repetitions per bit. Images must have enough capacity for the packet and redundancy; oversized messages return HTTP 400.

DWT extraction tests inverse quarter rotations and the app's default (+8,+8) translation, and validates recovered UTF-8 text using CRC32 without the original image or text. Tested square, portrait, and landscape PNGs recover after the default 90-degree rotation and translation. Re-embed DWT images made by earlier versions: the packet format has changed. This does not promise recovery for arbitrary angles, offsets, combined attacks, near-capacity messages, or destructive cropping. Strong JPEG compression and noise can still destroy packets. DCT extraction is unchanged: rotation typically fails, while translation may succeed depending on the image and packet position. Generated images use shared filenames in `static`, so this remains a single-user demonstration application.
