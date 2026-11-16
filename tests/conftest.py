import pytest

from app.store import reset_store


@pytest.fixture(autouse=True)
def clean_store():
    reset_store()
    yield
    reset_store()
