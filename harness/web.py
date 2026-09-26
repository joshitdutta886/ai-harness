"""Local web UI for the harness:  make ui  ->  http://localhost:8787

Watch runs live (a pipeline view, a step-by-step timeline, token counters and a diff),
replay finished runs, and start new runs on the practice tasks or your own repo.
Standard library only. Binds to 127.0.0.1, so it is never exposed to the network.

The UI only reads the files each run already writes (runs/*/trajectory.jsonl) and
starts runs as a normal `python3 -m harness` subprocess. It adds nothing to `make run`.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import ROOT

RUNS = os.path.join(ROOT, "runs")
RECORDINGS = os.path.join(ROOT, "examples", "recordings")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PROCS = {}  # run id -> Popen
LOCK = threading.Lock()


def _run_dirs():
    """Every folder under runs/ (up to 2 levels deep) that has a trajectory."""
    out = []
    if os.path.isdir(RUNS):
        for root, dirs, files in os.walk(RUNS):
            depth = root[len(RUNS):].count(os.sep)
            if depth > 2:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in ("repo", "demo_repo")]
            if "trajectory.jsonl" in files:
                out.append(root)
    return out


def _run_id(path: str) -> str:
    return os.path.relpath(path, RUNS).replace(os.sep, "~")


def _run_path(run_id: str) -> str:
    if run_id.startswith("rec:"):
        p = os.path.realpath(os.path.join(RECORDINGS, run_id[4:] + ".jsonl"))
        if not p.startswith(os.path.realpath(RECORDINGS) + os.sep):
            raise ValueError("bad id")
        return p
    p = os.path.realpath(os.path.join(RUNS, run_id.replace("~", os.sep), "trajectory.jsonl"))
    if not p.startswith(os.path.realpath(RUNS) + os.sep):
        raise ValueError("bad id")
    return p


def _read_events(path: str):
    events = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        break  # partially written last line: pick it up next poll
    except FileNotFoundError:
        pass
    return events


def _title(issue: str) -> str:
    for line in (issue or "").splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return (line[6:].strip() if line.lower().startswith("title:") else line)[:90]
    return "(no issue text)"


def _summary(run_id: str, path: str) -> dict:
    ev = _read_events(path)
    start = next((e for e in ev if e.get("kind") == "start"), {})
    end = next((e for e in reversed(ev) if e.get("kind") == "end"), None)
    with LOCK:
        proc = PROCS.get(run_id)
    alive = proc is not None and proc.poll() is None
    return {
        "id": run_id,
        "title": _title(start.get("issue", "")),
        "model": start.get("model", "?"),
        "started": ev[0]["t"] if ev else os.path.getmtime(path),
        "status": end["status"] if end else ("running" if alive else "incomplete"),
        "steps": end.get("steps") if end else max([e.get("step", 0) for e in ev if e.get("kind") == "step"] or [0]),
        "recording": run_id.startswith("rec:"),
    }


def list_runs():
    items = [_summary(_run_id(d), os.path.join(d, "trajectory.jsonl")) for d in _run_dirs()]
    if os.path.isdir(RECORDINGS):
        for fn in sorted(os.listdir(RECORDINGS)):
            if fn.endswith(".jsonl"):
                items.append(_summary("rec:" + fn[:-6], os.path.join(RECORDINGS, fn)))
    items.sort(key=lambda r: (not r["recording"], r["started"]), reverse=True)
    return items


def list_tasks():
    tasks = [{"name": "demo", "label": "Demo: cart discount bug", "repo": "examples/sample_repo",
              "issue": "examples/sample_issue.md", "test_cmd": ""}]
    try:
        with open(os.path.join(ROOT, "examples", "tasks.json"), "r", encoding="utf-8") as f:
            for t in json.load(f):
                t["label"] = "Practice: " + t["name"]
                tasks.append(t)
    except Exception:
        pass
    return tasks


def start_run(body: dict) -> dict:
    if not os.environ.get("AI_API_KEY"):
        return {"error": "AI_API_KEY is not set. Stop the UI (Ctrl+C), run: export AI_API_KEY=\"...\" and start make ui again."}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS, "ui-" + stamp)
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({"NO_COLOR": "1", "HARNESS_RUN_DIR": run_dir, "PYTHONUNBUFFERED": "1"})
    for k in ("REPO", "ISSUE", "ISSUE_FILE", "TEST_CMD"):
        env.pop(k, None)
    task_name = body.get("task") or ""
    if task_name and task_name != "custom":
        task = next((t for t in list_tasks() if t["name"] == task_name), None)
        if task is None:
            return {"error": "unknown task"}
        repo = os.path.join(run_dir, "repo")
        shutil.copytree(os.path.join(ROOT, task["repo"]), repo)  # never modify the examples
        with open(os.path.join(ROOT, task["issue"]), "r", encoding="utf-8") as f:
            env["ISSUE"] = f.read()
        env["REPO"] = repo
        if task.get("test_cmd"):
            env["TEST_CMD"] = task["test_cmd"]
    else:
        if not (body.get("repo") or "").strip() or not (body.get("issue") or "").strip():
            return {"error": "Give both a repository path (or git URL) and an issue."}
        env["REPO"] = body["repo"].strip()
        env["ISSUE"] = body["issue"]
        if (body.get("test_cmd") or "").strip():
            env["TEST_CMD"] = body["test_cmd"].strip()
    log = open(os.path.join(run_dir, "console.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "harness"], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    run_id = _run_id(run_dir)
    with LOCK:
        PROCS[run_id] = proc
    return {"id": run_id}


def stop_run(run_id: str) -> dict:
    with LOCK:
        proc = PROCS.get(run_id)
    if proc and proc.poll() is None:
        proc.terminate()
        return {"ok": True}
    return {"ok": False}


def console_tail(run_id: str) -> str:
    if run_id.startswith("rec:"):
        return ""
    p = os.path.join(RUNS, run_id.replace("~", os.sep), "console.log")
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[-3000:]
    except OSError:
        return ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/runs":
                return self._send(200, list_runs())
            if u.path == "/api/tasks":
                return self._send(200, list_tasks())
            if u.path == "/api/info":
                from .config import load_config
                info = {"key_set": bool(os.environ.get("AI_API_KEY"))}
                try:
                    cfg = load_config(probe=False) if info["key_set"] else None
                    info.update({"model": cfg.model, "provider": cfg.provider} if cfg else {})
                except BaseException:
                    pass
                return self._send(200, info)
            if u.path == "/api/events":
                run_id = q.get("run", [""])[0]
                since = int(q.get("since", ["0"])[0])
                ev = _read_events(_run_path(run_id))
                with LOCK:
                    proc = PROCS.get(run_id)
                alive = proc is not None and proc.poll() is None
                done = any(e.get("kind") == "end" for e in ev) or not alive
                return self._send(200, {"events": ev[since:], "next": len(ev), "done": done, "alive": alive,
                                        "console": "" if ev else console_tail(run_id)})
        except ValueError:
            return self._send(400, {"error": "bad request"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("content-length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        if u.path == "/api/start":
            return self._send(200, start_run(body))
        if u.path == "/api/stop":
            return self._send(200, stop_run(body.get("run", "")))
        self._send(404, {"error": "not found"})


def main() -> int:
    port = int(os.environ.get("HARNESS_UI_PORT", "8787"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://localhost:%d" % port
    print("AI Coding Harness UI running at %s   (Ctrl+C to stop)" % url, flush=True)
    if not os.environ.get("HARNESS_UI_NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for p in list(PROCS.values()):
            if p.poll() is None:
                p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
