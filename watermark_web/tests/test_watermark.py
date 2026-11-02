import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dct import embed_message_dct, extract_message_dct
from dwt import embed_message_dwt, extract_message_dwt
from app import app


@pytest.mark.parametrize('embed,extract', [(embed_message_dct, extract_message_dct), (embed_message_dwt, extract_message_dwt)])
@pytest.mark.parametrize('shape,message', [((257, 519), 'Hello مرحبا 🌊'), ((1024, 1024), 'x' * 260)])
def test_png_roundtrip(embed, extract, shape, message):
    image = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
    result = embed(image, message)
    assert result.shape == image.shape
    ok, png = cv2.imencode('.png', result)
    assert ok
    assert extract(cv2.imdecode(png, cv2.IMREAD_GRAYSCALE)) == message


@pytest.mark.parametrize('embed', [embed_message_dct, embed_message_dwt])
def test_capacity(embed):
    with pytest.raises(ValueError, match='too large'):
        embed(np.zeros((16, 16), dtype=np.uint8), 'Hello')


@pytest.mark.parametrize('method', ['dct', 'dwt'])
def test_web_workflow(tmp_path, monkeypatch, method):
    import app as web
    monkeypatch.setattr(web, 'STATIC_DIR', str(tmp_path))
    image = np.random.default_rng(7).integers(20, 235, (257, 519), dtype=np.uint8)
    _, png = cv2.imencode('.png', image)
    client = app.test_client()
    response = client.post('/embed', data={'image': (io.BytesIO(png.tobytes()), 'cover.png'), 'watermark': 'Test مرحبا', 'method': method})
    assert response.status_code == 200
    result = (tmp_path / 'embedded.png').read_bytes()
    response = client.post('/extract', data={'image': (io.BytesIO(result), 'marked.png')})
    assert response.status_code == 200
    assert 'Test مرحبا' in response.get_data(as_text=True)
    for attack in ['jpeg', 'blur', 'noise', 'rotation', 'translation']:
        response = client.post('/attack', data={'attack_type': attack})
        assert response.status_code == 200


@pytest.mark.parametrize('extension', ['.png', '.jpg', '.bmp', '.webp'])
def test_upload_creates_storage_and_accepts_desktop_images(tmp_path, monkeypatch, extension):
    import app as web
    storage = tmp_path / 'new' / 'static'
    monkeypatch.setattr(web, 'STATIC_DIR', str(storage))
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(extension, image)
    client = app.test_client()
    response = client.post('/embed', data={
        'image': (io.BytesIO(encoded.tobytes()), 'desktop' + extension),
        'watermark': 'Hello', 'method': 'dwt'})
    assert response.status_code == 200
    assert (storage / 'original.png').read_bytes().startswith(b'\x89PNG')
    assert (storage / 'embedded.png').exists()


def test_extract_first_on_fresh_checkout(tmp_path, monkeypatch):
    import app as web
    storage = tmp_path / 'missing-static'
    monkeypatch.setattr(web, 'STATIC_DIR', str(storage))
    image = embed_message_dct(np.full((256, 256), 128, dtype=np.uint8), 'Hello')
    _, encoded = cv2.imencode('.png', image)
    response = app.test_client().post('/extract', data={
        'image': (io.BytesIO(encoded.tobytes()), 'download.png'), 'method': 'dct'})
    assert response.status_code == 200
    assert 'Hello' in response.get_data(as_text=True)


def test_invalid_upload(tmp_path, monkeypatch):
    import app as web
    monkeypatch.setattr(web, 'STATIC_DIR', str(tmp_path))
    response = app.test_client().post('/embed', data={
        'image': (io.BytesIO(b'not an image'), 'bad.png'),
        'watermark': 'Hello', 'method': 'dwt'})
    assert response.status_code == 400
