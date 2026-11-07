from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid():
    r = client.post("/predict", json={"first": "Charizard", "second": "Blastoise"})
    assert r.status_code == 200
    body = r.json()
    assert body["winner"] in ("Charizard", "Blastoise")
    assert 0 <= body["probability_first_wins"] <= 1


def test_case_insensitive_names():
    r = client.post("/predict", json={"first": "charizard", "second": "BLASTOISE"})
    assert r.status_code == 200


def test_unknown_pokemon_returns_404():
    r = client.post("/predict", json={"first": "Fakemon", "second": "Blastoise"})
    assert r.status_code == 404


def test_missing_field_returns_422():
    r = client.post("/predict", json={"first": "Charizard"})
    assert r.status_code == 422


def test_swapping_order_is_consistent():
    a = client.post("/predict", json={"first": "Charizard", "second": "Blastoise"}).json()
    b = client.post("/predict", json={"first": "Blastoise", "second": "Charizard"}).json()
    assert abs(a["probability_first_wins"] + b["probability_first_wins"] - 1) < 0.2

def test_home_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Duelcast" in r.text


def test_pokemon_list():
    r = client.get("/pokemon")
    assert r.status_code == 200
    assert "Charizard" in r.json()