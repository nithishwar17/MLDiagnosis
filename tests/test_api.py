from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    # Even if models aren't loaded in the CI pipeline, it should return a 503 or 200 properly JSON formatted
    assert response.status_code in [200, 503]
    
def test_predict_endpoint_no_file():
    response = client.post("/predict")
    assert response.status_code == 422 # Unprocessable Entity (Missing file)
