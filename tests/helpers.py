import os
import shutil
import tempfile

from harness.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "examples", "sample_repo")


def sample_copy():
    d = tempfile.mkdtemp(prefix="harness_test_")
    dest = os.path.join(d, "repo")
    shutil.copytree(SAMPLE, dest)
    return dest


def fake_config(tmp_runs):
    c = Config(api_key="test", provider="deepseek", model="fake-model", runs_dir=tmp_runs)
    c.max_steps = 15
    return c.resolve(probe=False)
