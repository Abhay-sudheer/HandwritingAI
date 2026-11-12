from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from backend.app import main

class FakeRecognizer:
    provider='local'; device='cpu'; ready=True; error=None
    def __init__(self): self.model_path='test-model'
    def recognize_line(self, image): return 'recognized sample text'

def png_image():
    image=Image.new('RGB',(500,120),'white'); out=BytesIO(); image.save(out,format='PNG'); return out.getvalue()

@pytest.fixture(autouse=True)
def isolated_api_state(tmp_path,monkeypatch):
    monkeypatch.setattr(main,'Recognizer',FakeRecognizer)
    monkeypatch.setattr(main,'DB',tmp_path/'history.db')

def test_health_and_model_status():
    with TestClient(main.app) as client:
        assert client.get('/api/health').status_code==200
        assert 'model_available' in client.get('/api/model-status').json()

def test_invalid_upload_type():
    with TestClient(main.app) as client:
        assert client.post('/api/recognize',files={'file':('file.txt',b'hello','text/plain')}).status_code==415

def test_history_clear_and_missing_delete():
    with TestClient(main.app) as client:
        assert client.delete('/api/history').status_code==200
        assert client.delete('/api/history/missing-record').status_code==404
        assert client.get('/api/history').json()==[]

def test_valid_upload_and_history_operations(tmp_path,monkeypatch):
    monkeypatch.setattr(main,'Recognizer',FakeRecognizer)
    monkeypatch.setattr(main,'DB',tmp_path/'history.db')
    with TestClient(main.app) as client:
        response=client.post('/api/recognize',files={'file':('note.png',png_image(),'image/png')})
        assert response.status_code==200
        item=response.json()
        assert item['text']=='recognized sample text'
        assert item['status']=='success'
        assert client.get('/api/history').json()[0]['filename']=='note.png'
        assert client.delete('/api/history/'+item['id']).json()['deleted'] is True
        assert client.get('/api/history').json()==[]
