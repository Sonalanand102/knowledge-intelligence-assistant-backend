from backend.app.core.config import settings


def test_settings_load_required_configuration():
    assert settings.database_url
    assert settings.redis_url
    assert settings.qdrant_url
    assert settings.qdrant_collection_name
    assert settings.gemini_api_key