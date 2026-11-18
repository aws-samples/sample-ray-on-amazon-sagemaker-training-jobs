"""Tests for running Bash entry scripts from launcher.py.

They run on a laptop: the SageMaker / Ray / AWS imports of launcher.py are
replaced with stubs, and only the Bash execution path is exercised.
"""

import importlib.util
import pathlib
import sys
import types

import pytest

LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "launcher.py"


@pytest.fixture
def launcher(monkeypatch):
    stubs = {
        "boto3": types.ModuleType("boto3"),
        "requests": types.ModuleType("requests"),
        "yaml": types.ModuleType("yaml"),
        "ray": types.ModuleType("ray"),
        "sagemaker_training": types.ModuleType("sagemaker_training"),
        "sagemaker_training.environment": types.ModuleType("sagemaker_training.environment"),
        "botocore": types.ModuleType("botocore"),
        "botocore.exceptions": types.ModuleType("botocore.exceptions"),
    }
    stubs["botocore.exceptions"].ClientError = type("ClientError", (Exception,), {})
    stubs["botocore.exceptions"].NoCredentialsError = type("NoCredentialsError", (Exception,), {})
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("launcher_under_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_script(tmp_path, body):
    script = tmp_path / "run.sh"
    script.write_text(body)
    return script


def test_unknown_args_are_forwarded_to_bash_script(launcher, tmp_path, capfd):
    script = write_script(tmp_path, 'echo "ARGS:$*"\n')
    launcher.entry_script_args = ["--epochs", "10", "--lr", "1e-4"]
    launcher._execute_bash_script(str(script), str(tmp_path), {"PATH": "/usr/bin:/bin"})
    assert "ARGS:--epochs 10 --lr 1e-4" in capfd.readouterr().out


def test_no_args_by_default(launcher, tmp_path, capfd):
    script = write_script(tmp_path, 'echo "COUNT:$#"\n')
    launcher._execute_bash_script(str(script), str(tmp_path), {"PATH": "/usr/bin:/bin"})
    assert "COUNT:0" in capfd.readouterr().out


def test_output_is_streamed_not_buffered_until_exit(launcher, tmp_path):
    # The script writes a marker, then waits until the test has seen it.
    # If the launcher captured the output, the marker would only appear after
    # the script exits, and the script would never exit (the test would hang
    # until the timeout below fails it).
    marker = tmp_path / "seen"
    script = write_script(
        tmp_path,
        f'echo STARTED\nfor i in $(seq 100); do [ -f "{marker}" ] && exit 0; sleep 0.05; done\nexit 7\n',
    )
    log = tmp_path / "stdout.log"
    import subprocess
    import threading

    with open(log, "w") as out:
        real_run = subprocess.run

        def run_to_file(*a, **kw):
            assert "capture_output" not in kw and "stdout" not in kw
            return real_run(*a, stdout=out, stderr=out, **kw)

        launcher.subprocess.run = run_to_file
        try:
            t = threading.Thread(
                target=launcher._execute_bash_script,
                args=(str(script), str(tmp_path), {"PATH": "/usr/bin:/bin"}),
            )
            t.start()
            for _ in range(100):
                if "STARTED" in log.read_text():
                    marker.touch()
                    break
                t.join(0.05)
            t.join(10)
        finally:
            launcher.subprocess.run = real_run
    assert marker.exists(), "script output was not visible while the script was running"


def test_failure_still_raises(launcher, tmp_path):
    script = write_script(tmp_path, "echo boom >&2\nexit 3\n")
    with pytest.raises(launcher.subprocess.CalledProcessError) as e:
        launcher._execute_bash_script(str(script), str(tmp_path), {"PATH": "/usr/bin:/bin"})
    assert e.value.returncode == 3


def test_launcher_options_are_not_forwarded(launcher, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["launcher.py", "-e", "run.sh", "--wait-shutdown", "5", "--epochs", "10"]
    )
    args, unknown = launcher._parse_args()
    assert args.wait_shutdown == 5
    assert unknown == ["--epochs", "10"]
