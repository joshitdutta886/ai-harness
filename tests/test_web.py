"""The optional web UI server (no network, no model)."""
import json
import os
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from harness import web


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def get(self, path):
        with self.opener.open("http://127.0.0.1:%d%s" % (self.port, path), timeout=10) as r:
            return r.read()

    def test_page_and_tasks(self):
        self.assertIn(b"AI Coding Harness", self.get("/"))
        names = [t["name"] for t in json.loads(self.get("/api/tasks"))]
        self.assertEqual(names[0], "demo")
        self.assertIn("median", names)

    def test_recording_listed_and_readable(self):
        runs = json.loads(self.get("/api/runs"))
        rec = [r for r in runs if r["recording"]]
        self.assertTrue(rec, "bundled recording should be listed")
        ev = json.loads(self.get("/api/events?run=" + rec[0]["id"] + "&since=0"))
        kinds = [e["kind"] for e in ev["events"]]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "end")

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            web._run_path("..~..~etc")
        with self.assertRaises(ValueError):
            web._run_path("rec:../../x")

    def test_start_requires_key(self):
        old = os.environ.pop("AI_API_KEY", None)
        try:
            self.assertIn("AI_API_KEY", web.start_run({"task": "demo"})["error"])
        finally:
            if old is not None:
                os.environ["AI_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
