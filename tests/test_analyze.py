import pytest

VALID_PAYLOAD = {
    "startup_name": "Acme Corp",
    "website": "https://acme.example.com",
    "description": "A revolutionary SaaS platform for enterprise workflow automation.",
    "pitch_deck_url": "https://docs.acme.example.com/pitch.pdf",
    "founder_linkedin_urls": ["https://linkedin.com/in/janedoe"],
}


@pytest.mark.anyio
async def test_analyze_success(client):
    response = await client.post("/api/v1/analyze", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["startup_name"] == "Acme Corp"
    assert 0 <= body["venture_score"] <= 100
    assert 0 <= body["market_score"] <= 100
    assert 0 <= body["founder_score"] <= 100
    assert 0 <= body["traction_score"] <= 100
    assert 0 <= body["confidence"] <= 1
    assert isinstance(body["recommendations"], list)


@pytest.mark.anyio
async def test_analyze_minimal_payload(client):
    minimal = {
        "startup_name": "Minimal Co",
        "website": "https://minimal.example.com",
        "description": "A short description for the startup.",
    }
    response = await client.post("/api/v1/analyze", json=minimal)
    assert response.status_code == 200
    assert response.json()["startup_name"] == "Minimal Co"


@pytest.mark.anyio
async def test_analyze_missing_required_field(client):
    incomplete = {"startup_name": "Missing Fields Inc"}
    response = await client.post("/api/v1/analyze", json=incomplete)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_analyze_empty_name_rejected(client):
    payload = {
        "startup_name": "",
        "website": "https://empty.example.com",
        "description": "A description of the startup.",
    }
    response = await client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_analyze_short_description_rejected(client):
    payload = {
        "startup_name": "Short Desc Co",
        "website": "https://short.example.com",
        "description": "Too short",
    }
    response = await client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_analyze_invalid_website_rejected(client):
    payload = {
        "startup_name": "Bad URL Co",
        "website": "not-a-url",
        "description": "A perfectly valid startup description for testing.",
    }
    response = await client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422
