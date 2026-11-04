"""Single-user loopback HTTP interface with explicit mutating requests."""

import copy
import json
import secrets
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from siliconshield.web_demo.backend import (
    ROUTING_SCOPES,
    Demo,
    read_json,
    state_text,
)

ROOT = Path(__file__).resolve().parent


class DemoServer(ThreadingHTTPServer):
    """Keep application state scoped to one bound server."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], demo: Demo) -> None:
        self.demo = demo
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    """Expose only explicit assets and bounded assessment/study actions."""

    server: DemoServer
    server_version = "SiliconShieldDemo/1.0"

    def setup(self) -> None:
        """Bound incomplete local requests without affecting model jobs."""
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, message: str, *args: Any) -> None:
        """Keep request bodies and RTL out of HTTP access logs."""

    def send(
        self, code: int, body: Any, content_type: str = "application/json"
    ) -> None:
        """Send local-only content with no executable third-party assets."""
        if content_type == "application/json":
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def local_host(self) -> bool:
        """Reject foreign Host headers, including DNS rebinding requests."""
        port = self.server.server_port
        return self.headers.get("Host", "") in {
            f"127.0.0.1:{port}",
            f"localhost:{port}",
        }

    def do_GET(self) -> None:
        """Serve the UI and read-only receipts from this session."""
        if (
            not self.local_host()
            or self.headers.get("Sec-Fetch-Site") == "cross-site"
        ):
            self.send(403, {"error": "Localhost access only."})
            return
        path = urlparse(self.path).path
        demo = self.server.demo
        try:
            if path == "/api/bootstrap":
                gpu = demo.config["gpu"]
                devices = [{"id": "cpu", "label": "CPU · server"}]
                if gpu is not None:
                    devices.append({"id": f"cuda:{gpu}", "label": f"GPU {gpu}"})
                self.send(
                    200,
                    {
                        "token": self.server.token,
                        "examples": read_json(
                            ROOT / "examples.example.json", []
                        ),
                        "title": "SiliconShield",
                        "gpu": gpu,
                        "devices": devices,
                        "default_device": demo.inference_device(),
                        "toy_updates": 3,
                        "training_enabled": demo.config["training_enabled"],
                        "glimmer_configured": bool(demo.config["glimmer"]),
                    },
                )
                return
            if path == "/api/status":
                self.send(200, demo.status())
                return
            if path in {"/api/training", "/api/training/receipt"}:
                self.send(200, demo.training_status())
                return
            if path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                if len(job_id) != 32 or any(
                    c not in "0123456789abcdef" for c in job_id
                ):
                    self.send(404, {"error": "Unknown request."})
                    return
                with demo.lock:
                    value = copy.deepcopy(demo.jobs.get(job_id))
                if value is None:
                    value = read_json(demo.state / f"request-{job_id}.json")
                if value:
                    self.send(200, value)
                elif demo.was_evicted(job_id):
                    self.send(
                        410, {"error": "Request expired; resubmit the task."}
                    )
                else:
                    self.send(404, {"error": "Unknown request."})
                return
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
                "/favicon.svg": ("favicon.svg", "image/svg+xml"),
            }
            if path not in assets:
                self.send(404, {"error": "Not found."})
                return
            name, mime = assets[path]
            self.send(200, (ROOT / "static" / name).read_bytes(), mime)
        except Exception as exc:
            # HTTP boundary preserves infrastructure failures as execution errors.
            self.send(500, {"error": str(exc)})

    def do_POST(self) -> None:
        """Authorize mutations before dispatching any model or training work."""
        port = self.server.server_port
        allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
        token = self.headers.get("X-Demo-Token", "")
        try:
            authorized = secrets.compare_digest(token, self.server.token)
        except (TypeError, ValueError):
            authorized = False
        if (
            not self.local_host()
            or self.headers.get("Origin") not in allowed | {None}
            or not authorized
        ):
            self.send(
                403, {"error": "Invalid local session. Reload this page."}
            )
            return
        try:
            if (
                self.headers.get("Transfer-Encoding") is not None
                or self.headers.get_content_type() != "application/json"
            ):
                raise ValueError(
                    "A bounded application/json request is required."
                )
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 128000:
                raise ValueError("Request size exceeds the demo limit.")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("JSON object required.")
            result = self.dispatch(urlparse(self.path).path, body)
            self.send(202 if "job_id" in result else 200, result)
        except (ValueError, TypeError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception as exc:
            self.send(500, {"error": str(exc)})

    def dispatch(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Keep paths, commands, and training settings out of browser authority."""
        demo = self.server.demo
        if path in {"/api/models/load", "/api/assess", "/api/models/unload"}:
            model = body.get("model")
            if not isinstance(model, str) or model not in demo.entries:
                raise ValueError(
                    "Register this model in the local model registry first."
                )
            if path.endswith("/unload"):
                return demo.unload(model)
            device = demo.inference_device(body.get("device"))
            if path.endswith("/load"):
                return demo.submit_task(
                    "load", lambda log: demo.load(model, log, device)
                )
            state_text(body)
            return demo.submit_task(
                "assessment", lambda log: demo.assess(body, log)
            )
        if path == "/api/route":
            state_text(body)
            if body.get("scope") not in ROUTING_SCOPES:
                raise ValueError("Select a measured routing domain.")
            if not demo.config["registry"]:
                raise ValueError("Configure a model registry before routing.")
            return demo.submit_task("router", lambda log: demo.route(body, log))
        if path == "/api/training/start":
            if set(body) != {"confirm"} or body.get("confirm") is not True:
                raise ValueError(
                    "Only an explicit start confirmation is accepted."
                )
            if not demo.config["training_enabled"]:
                raise ValueError(
                    "Toy training is disabled in the local configuration."
                )
            with demo.lock:
                if any(
                    j["kind"] == "training-launch" and j["status"] == "running"
                    for j in demo.jobs.values()
                ):
                    raise ValueError(
                        "A training launch is already in progress."
                    )
                return demo.submit_task(
                    "training-launch", demo.prepare_training
                )
        if path == "/api/training/stop":
            return demo.stop_training()
        raise ValueError("Unknown action.")


def serve(config: dict[str, Any]) -> None:
    """Own one local session while training jobs retain their durable lifecycle."""
    import fcntl

    state = Path(config["state_dir"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / "server.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit(
                "This demo state directory already has a server."
            ) from exc
        demo = Demo(config)
        try:
            with DemoServer(("127.0.0.1", config["port"]), demo) as server:

                def stop(_signum: int, _frame: Any) -> None:
                    threading.Thread(
                        target=server.shutdown, daemon=True
                    ).start()

                signal.signal(signal.SIGTERM, stop)
                print(
                    f"SiliconShield demo: http://127.0.0.1:{server.server_port}",
                    flush=True,
                )
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
        finally:
            demo.close()
