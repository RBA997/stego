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

Choose DCT or DWT on Embed, then upload the downloaded PNG on Extract in the same browser session. API clients may also send `method=dct` or `method=dwt` to `/extract`. DCT uses complete 8×8 blocks and preserves remaining edge pixels. DWT uses Haar detail-coefficient pairs. Both encode UTF-8 text with a 16-bit payload length and five repetitions per bit. Images must have enough capacity for the packet and redundancy; oversized messages return HTTP 400.

Attack results measure degradation; these algorithms do not guarantee recovery after strong JPEG compression, noise, rotation, or translation. Generated images use shared filenames in `static`, so this remains a single-user demonstration application.
