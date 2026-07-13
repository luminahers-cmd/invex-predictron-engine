from app.adapters.predictron_adapter import _mock_response
from app.schemas.analysis import StartupAnalysisRequest


def test_mock_response_structure():
    request = StartupAnalysisRequest(
        startup_name="Test Startup",
        website="https://test.example.com",
        description="A test startup description for mock response validation.",
    )
    result = _mock_response(request)

    assert result.startup_name == "Test Startup"
    assert 0 <= result.venture_score <= 100
    assert 0 <= result.market_score <= 100
    assert 0 <= result.founder_score <= 100
    assert 0 <= result.traction_score <= 100
    assert 0 <= result.confidence <= 1
    assert isinstance(result.recommendations, list)
    assert len(result.recommendations) > 0
