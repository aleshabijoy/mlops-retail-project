import pytest
from celery import current_app

@pytest.fixture(autouse=True)
def set_celery_eager():
    # Set Celery to eager mode for all tests
    current_app.conf.task_always_eager = True
    current_app.conf.task_eager_propagates = True
