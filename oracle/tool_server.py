"""Tool API server — runs on the Oracle VM (port 8766), stdlib only.

Same logic as the Colab version: the Mac client's execute_tool POSTs here,
the real work (background claim lookup) happens on this box.
"""
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_jobs = {}


def _run_backend_job(search_id, claim_number):
    time.sleep(10)  # the slow claims backend
    _jobs[search_id] = {
        "done": True,
        "result": {
            "claim_number": claim_number,
            "status": "In review",
            "eta_days": 3,
            "last_update": "Assigned to adjuster Priya Nair",
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/run-tool":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            req = {}
        name = req.get("name")
        args = req.get("arguments", {}) or {}
        if name == "lookup_claim_status":
            search_id = "SRCH-" + uuid.uuid4().hex[:6]
            _jobs[search_id] = {"done": False}
            threading.Thread(
                target=_run_backend_job,
                args=(search_id, args.get("claim_number", "unknown")),
                daemon=True,
            ).start()
            self._send({"search_id": search_id, "status": "pending", "eta_seconds": 10})
        elif name == "get_claim_result":
            job = _jobs.get(args.get("search_id"))
            if not job:
                self._send({"status": "error", "note": "unknown search_id"})
            elif not job["done"]:
                self._send({"status": "still_pending",
                            "note": "Tell the user it is still running; offer to check again shortly."})
            else:
                self._send({"status": "complete", **job["result"]})
        else:
            self._send({"status": "error", "note": "unknown tool: " + str(name)})

    def log_message(self, *args):
        pass


print("tool server starting on 127.0.0.1:8766", flush=True)
ThreadingHTTPServer(("127.0.0.1", 8766), Handler).serve_forever()
