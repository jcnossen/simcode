"""pytest options of the SIMCODE tests:
  --work4pi <dir>   the 4Pi work folder (psf4pi/NPC_f_dxy15.json, models/...): the 4Pi tests that need its files run
                    (they are skipped without it)"""
import pytest


def pytest_addoption(parser):
    parser.addoption('--work4pi', default=None, help='the 4Pi work folder (the 4Pi tests that need its files are skipped without it)')


@pytest.fixture
def work4pi(request):
    return request.config.getoption('--work4pi')
