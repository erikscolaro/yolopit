import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent / "scripts"


def run_script(name: str, out: Path, timeout: int = 3600) -> str:
    """Run tests/scripts/<name> in its own process (the scripts train, patch globals and print a
    check list): passes if it exits 0 and prints ALL CHECKS PASSED."""
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "YOLOPIT_TEST_OUT": str(out)}
    r = subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=out, env=env,
                       capture_output=True, text=True, timeout=timeout)
    log = r.stdout + r.stderr
    (out / f"{name}.log").write_text(log)
    assert r.returncode == 0, f"{name} exited with {r.returncode}:\n{log[-4000:]}"
    assert "ALL CHECKS PASSED" in r.stdout, f"{name}: some checks failed:\n{log[-4000:]}"
    return log


@pytest.fixture
def script(tmp_path):
    return lambda name, **kw: run_script(name, tmp_path, **kw)
