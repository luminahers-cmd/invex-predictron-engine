import pytest

from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.startup import Startup


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def sample_startup() -> Startup:
    return Startup(
        name="SampleCo",
        website="https://sampleco.example.com",
        description="A sample startup for testing purposes.",
        pitch_deck_url=None,
        founder_linkedin_urls=[],
        raw_data={},
    )


@pytest.fixture
def sample_collected_data() -> CollectedData:
    return CollectedData(
        startup_name="SampleCo",
        website_domain="sampleco.example.com",
        description_tokens=["sample", "startup", "testing", "purposes"],
        description_word_count=4,
        has_website=True,
        has_pitch_deck=False,
        founder_count=0,
        url_metadata={"website_url": "https://sampleco.example.com"},
        enrichment_signals={},
    )
