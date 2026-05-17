import threading
import time

from auroch_syna.worldgen.backends.rpc_client import ModelClientRPC


def test_rpc_ping():
    # Start the mock server in a background thread
    from auroch_syna.worldgen.backends.mock_server import run_mock_server

    server_thread = threading.Thread(target=run_mock_server, kwargs={"host": "127.0.0.1", "port": 8001}, daemon=True)
    server_thread.start()
    time.sleep(0.2)

    client = ModelClientRPC("http://127.0.0.1:8001")
    r = client.ping()
    assert r.get("status") == "ok"
