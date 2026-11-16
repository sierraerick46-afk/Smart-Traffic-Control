import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The tests exercise the web system and the saved scikit-learn model. PyTorch is only needed for the DQN
# simulator, so a stand-in is used when it is not installed.
try:
    import torch  # noqa: F401
except ImportError:
    import torch_stub
    torch_stub.install()

os.environ["SKIP_APP_AT_IMPORT"] = "1"
os.environ["TEST_USERS_PASSWORD"] = "Test12345"
os.environ["ERICK_PASSWORD"] = "Erick12345"
os.environ["SECRET_KEY"] = "test-secret"

import pytest  # noqa: E402

from models.database import Database  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    import app as app_module
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    application = app_module.create_app(db=Database(url="", path=str(db_path)), light=True)
    application.config["TESTING"] = True
    application.config["DB_PATH"] = str(db_path)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


def csrf(client):
    """Reads the token the server put in the session when it rendered a page with a form."""
    client.get("/login")
    with client.session_transaction() as s:
        have = "csrf" in s
    if not have:
        client.get("/crisp-ml")      # logged-in users get the token through the log-out form of the navigation
    with client.session_transaction() as s:
        return s["csrf"]


def login(client, username, password="Test12345"):
    token = csrf(client)
    return client.post("/login", data={"username": username, "password": password, "csrf_token": token})


def post(client, url, data=None, **kw):
    data = dict(data or {})
    data["csrf_token"] = csrf(client)
    return client.post(url, data=data, **kw)
