import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag import build_assistant  # noqa: E402


@pytest.fixture(scope="session")
def assistant():
    return build_assistant()


@pytest.fixture(scope="session")
def kg(assistant):
    return assistant.kg


@pytest.fixture(scope="session")
def extraction(assistant):
    return assistant.extraction
