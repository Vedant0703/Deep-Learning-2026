"""Run training on a Colab GPU from a local terminal (Cursor, Claude Code, ...) with the Google Colab CLI.

One-time setup:  uv tool install google-colab-cli   (then run `colab usage` once to sign in)

    python scripts/colab.py up                      start a GPU session, upload code and data
    python scripts/colab.py sync                    upload the current code (rerun after every edit)
    python scripts/colab.py train RUN [ARGS...]     start training in the background on the VM, then follow its log
    python scripts/colab.py screen [EPOCHS]         train every candidate model concurrently (default 40 epochs)
    python scripts/colab.py follow RUN [RUN...]     stream the log(s) of running jobs until they finish
    python scripts/colab.py status RUN [RUN...]     latest epochs of each run (works while the kernel is busy)
    python scripts/colab.py fetch RUN [RUN...]      download logs and weights to runs/RUN/
    python scripts/colab.py down                    stop the session (stops compute-unit billing)

ARGS after RUN are passed to `python -m src.train`, e.g.
    python scripts/colab.py train final_a --model resnet_a --epochs 200

Training runs as a detached process on the VM, so stopping `follow` (Ctrl-C, closing the terminal)
does not stop training; rerun `follow RUN` to reattach. Every run uses --resume, so rerunning
`train RUN` after a crash continues from the last checkpoint.

Environment: COLAB_SESSION (default dl1), COLAB_GPU (default A100).
"""
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSION = os.environ.get("COLAB_SESSION", "dl1")
GPU = os.environ.get("COLAB_GPU", "A100")
REMOTE = "/content/proj"
MODELS = ("resnet_a", "resnet_b", "resnet_c")
FETCH_FILES = ("config.json", "history.json", "train.log", "final.pt", "best.pt")


def colab(*args, check=True):
    cmd = ["colab", *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check).returncode


def remote_python(code, timeout=600):
    """Run Python source on the session's kernel (its stdout streams back here)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        colab("exec", "-s", SESSION, "--timeout", str(timeout), "-f", f.name)
    finally:
        os.unlink(f.name)


# Remote helper: start `python -m src.train` detached (own session, output to train.log, pid file).
LAUNCH = """
import os, subprocess, sys
os.chdir({remote!r})
for run, args in {jobs!r}:
    run_dir = f"runs/{{run}}"
    os.makedirs(run_dir, exist_ok=True)
    log = open(f"{{run_dir}}/train.log", "a")
    cmd = [sys.executable, "-u", "-m", "src.train", "--out", run_dir, "--resume", *args]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    open(f"{{run_dir}}/pid", "w").write(str(p.pid))
    print(f"[{{run}}] started pid {{p.pid}}: {{' '.join(cmd[2:])}}", flush=True)
"""

# Remote helper: print new log lines of each run until all of their processes have exited.
FOLLOW = """
import os, time
os.chdir({remote!r})
runs = {runs!r}

def finished(pid):
    try:
        return os.waitpid(pid, os.WNOHANG)[0] != 0   # our child: reap it once it exits
    except ChildProcessError:                        # not our child (e.g. kernel restarted)
        return not os.path.exists(f"/proc/{{pid}}")

state = {{}}
for run in runs:
    f = open(f"runs/{{run}}/train.log")
    lines = f.readlines()
    for line in lines[-5:]:
        print(f"[{{run}}] {{line}}" if len(runs) > 1 else line, end="", flush=True)
    state[run] = (f, int(open(f"runs/{{run}}/pid").read()))

while state:
    idle = True
    for run, (f, pid) in list(state.items()):
        line = f.readline()
        if line:
            idle = False
            print(f"[{{run}}] {{line}}" if len(runs) > 1 else line, end="", flush=True)
        elif finished(pid):
            print(f"[{{run}}] process exited", flush=True)
            del state[run]
    if idle:
        time.sleep(2)
"""


def cmd_up():
    colab("new", "-s", SESSION, "--gpu", GPU)
    remote_python(
        "import subprocess, os\n"
        f"os.makedirs({REMOTE!r}, exist_ok=True)\n"
        "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],"
        " capture_output=True, text=True).stdout, 'CPUs:', os.cpu_count())\n")
    cmd_sync()
    colab("upload", "-s", SESSION, str(ROOT / "data.zip"), f"{REMOTE}/data.zip")
    remote_python(
        "import os, zipfile, subprocess, sys\n"
        f"os.chdir({REMOTE!r})\n"
        "zipfile.ZipFile('data.zip').extractall('data'); os.remove('data.zip')\n"
        "r = subprocess.run([sys.executable, '-c', \"from src.data import load_split; "
        "[print(s, load_split('data', s)[0].shape) for s in ('train', 'test')]\"],"
        " capture_output=True, text=True)\n"
        "print(r.stdout, r.stderr[-2000:])\n", timeout=900)
    colab("usage", check=False)


def cmd_sync():
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "code.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(ROOT / "src", "src", filter=lambda t: None if "__pycache__" in t.name else t)
            tar.add(ROOT / "requirements.txt", "requirements.txt")
        colab("upload", "-s", SESSION, str(archive), f"{REMOTE}/code.tgz")
    remote_python(
        "import os, shutil, tarfile\n"
        f"os.chdir({REMOTE!r})\n"
        "shutil.rmtree('src', ignore_errors=True)\n"
        "tarfile.open('code.tgz').extractall('.', filter='data'); os.remove('code.tgz')\n"
        "print('synced:', sorted(os.listdir('src')))\n")


def cmd_train(run, *args):
    remote_python(LAUNCH.format(remote=REMOTE, jobs=[(run, list(args))]))
    cmd_follow(run)


def cmd_screen(epochs="40"):
    # One A100 is underused by a single 5M-parameter model, so the candidates share it.
    jobs = [(f"screen_{m}", ["--model", m, "--epochs", str(epochs), "--num-workers", "4"]) for m in MODELS]
    remote_python(LAUNCH.format(remote=REMOTE, jobs=jobs))
    cmd_follow(*[run for run, _ in jobs])


def cmd_follow(*runs):
    remote_python(FOLLOW.format(remote=REMOTE, runs=list(runs)), timeout=24 * 3600)


def cmd_status(*runs):
    for run in runs:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "history.json"
            if colab("download", "-s", SESSION, f"{REMOTE}/runs/{run}/history.json", str(local), check=False):
                print(f"[{run}] no history yet")
                continue
            hist = json.loads(local.read_text())
        last = hist[-1]
        msg = (f"[{run}] epoch {last['epoch']} | train loss {last['train_loss']:.4f} acc {last['train_acc']:.2f} "
               f"| {last['time']:.1f}s/epoch")
        if last.get("val_acc") is not None:
            msg += f" | val acc {last['val_acc']:.2f} (best {max(h['val_acc'] for h in hist):.2f})"
        print(msg)


def cmd_fetch(*runs):
    for run in runs:
        out = ROOT / "runs" / run
        out.mkdir(parents=True, exist_ok=True)
        for name in FETCH_FILES:
            if colab("download", "-s", SESSION, f"{REMOTE}/runs/{run}/{name}", str(out / name), check=False):
                print(f"  (skipped {name}: not on the VM)")
        print(f"[{run}] saved to {out}")


def cmd_down():
    colab("stop", "-s", SESSION)


COMMANDS = {"up": cmd_up, "sync": cmd_sync, "train": cmd_train, "screen": cmd_screen, "follow": cmd_follow,
            "status": cmd_status, "fetch": cmd_fetch, "down": cmd_down}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    start = time.time()
    COMMANDS[sys.argv[1]](*sys.argv[2:])
    print(f"done in {time.time() - start:.0f}s")
