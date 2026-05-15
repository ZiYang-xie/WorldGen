"""Tiny mock model server for local development.

Implements a few simple HTTP endpoints that return canned JSON responses.
This avoids pulling in FastAPI/Flask during early development.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Tuple


class MockModelHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if self.path.endswith("/ping") or self.path.endswith("ping"):
            return self._send_json({"status": "ok", "msg": payload.get("msg", "pong")})

        if self.path.endswith("/predict_depth") or self.path.endswith("predict_depth"):
            # Return a tiny depth stub
            return self._send_json({"status": "ok", "depth_shape": [64, 128], "note": "mock depth"})

        if self.path.endswith("/generate_pano") or self.path.endswith("generate_pano"):
            return self._send_json({"status": "ok", "pano_url": "http://localhost:8000/static/sample_pano.jpg"})

        if self.path.endswith("/inpaint") or self.path.endswith("inpaint"):
            return self._send_json({"status": "ok", "inpainted_url": "http://localhost:8000/static/inpainted.jpg"})

        if self.path.endswith("/segment") or self.path.endswith("segment"):
            return self._send_json({"status": "ok", "masks": ["fg", "bg"], "note": "mock segmentation"})

        if self.path.endswith("/sharp") or self.path.endswith("sharp"):
            return self._send_json({"status": "ok", "splats": 512, "note": "mock sharp output"})

        return self._send_json({"error": "unknown endpoint"}, code=404)


def run_mock_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), MockModelHandler)
    addr = server.server_address
    print(f"MockModelServer running at http://{addr[0]}:{addr[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down mock server")
        server.shutdown()


if __name__ == "__main__":
    run_mock_server()
