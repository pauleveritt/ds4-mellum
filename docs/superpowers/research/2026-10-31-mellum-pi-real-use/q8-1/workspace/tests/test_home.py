from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Come in. Sit down. Tell us about your human." in response.text
    # Additional check for navbar links maybe
    assert "Home" in response.text
    assert "Complaints" in response.text