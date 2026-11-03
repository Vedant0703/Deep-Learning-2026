"""Run training on a Colab GPU from a local terminal (Cursor, Claude Code, ...) with the Google Colab CLI.

One-time setup:  uv tool install google-colab-cli   (then run `colab usage` once to sign in)

    python scripts/colab.py up                      start a GPU session, upload code and data
    python scripts/colab.py sync                    upload the current code (rerun after every edit)
    python scripts/colab.py data                    (re)upload and unpack data.zip
    python scripts/colab.py put FILE [DEST]         upload a local file of any size, by default to the same path
                                                    on the VM (a teacher, or a backed-up runs/RUN/last.pt)
    python scripts/colab.py start RUN [ARGS...]     start training in the background on the VM
    python scripts/colab.py train RUN [ARGS...]     start, then follow its log
    python scripts/colab.py screen [EPOCHS]         train every candidate model concurrently (default 40 epochs)
    python scripts/colab.py follow RUN [RUN...]     print new log lines until the runs finish, then fetch them;
                                                    backs up each run's last.pt locally every 10 minutes
    python scripts/colab.py status RUN [RUN...]     latest epoch of each run, and whether the supervisor is alive
    python scripts/colab.py eval CKPT [CKPT...]     clean + brightness/contrast-shifted val accuracy (src.evaluate)
    python scripts/colab.py fetch RUN [RUN...]      download logs and weights to runs/RUN/
    python scripts/colab.py down                    stop the session (stops compute-unit billing)

ARGS after RUN are passed to `python -m src.train`, e.g.
    python scripts/colab.py train final_a --model resnet_a --epochs 200

How it works: Colab prunes a VM a few minutes after its kernel goes idle, even if background processes are
still training. So `up` starts a supervisor cell that keeps the kernel busy. Everything after that is handed
to the supervisor as a small job file uploaded into /content/proj/queue through the Contents API, which works
while the kernel is busy. Training runs as detached processes. `follow` and `status` only download files,
so stopping them (Ctrl-C, closing the terminal) does not affect the VM. The supervisor exits, and Colab then
releases the VM, after IDLE_EXIT seconds with no training running. Every run uses --resume.

The VM also needs this computer awake: when it slept (lid closed), Colab reclaimed the VM with all runs on it.
While the supervisor is connected, macOS idle sleep is blocked with caffeinate, but closing the lid still sleeps.
To continue a run on a new VM from its local backup:
    python scripts/colab.py put runs/RUN/last.pt && python scripts/colab.py start RUN [same ARGS]

Environment: COLAB_SESSION (default dl1), COLAB_GPU (default A100).
"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSION = os.environ.get("COLAB_SESSION", "dl1")
GPU = os.environ.get("COLAB_GPU", "A100")
REMOTE = "/content/proj"
MODELS = ("resnet_a", "resnet_b", "resnet_c")
FETCH_FILES = ("config.json", "history.json", "train.log", "final.pt", "best.pt")
IDLE_EXIT = 30 * 60
SUPERVISOR_LOG = ROOT / "runs" / f".supervisor-{SESSION}.log"


def colab(*args, check=True, timeout=None):
    cmd = ["colab", *args]
    print("$", " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, check=check, timeout=timeout).returncode
    except subprocess.TimeoutExpired:  # the CLI can hang after a failed websocket connect
        print(f"[colab.py] killed after {timeout}s", flush=True)
        return 124


def remote_python(code, timeout=600, attempts=3):
    """Run Python source on the kernel and stream its output. Only usable before the supervisor holds the kernel.

    `colab exec` sometimes fails to open its websocket (and then hangs), so it is killed and retried;
    `code` must be safe to run again.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        for attempt in range(1, attempts + 1):
            if colab("exec", "-s", SESSION, "--timeout", str(timeout), "-f", f.name,
                     check=False, timeout=timeout + 30) == 0:
                return
            if attempt < attempts:
                print(f"[colab.py] colab exec failed, retrying ({attempt}/{attempts - 1})", flush=True)
                time.sleep(10)
        sys.exit(f"colab exec failed {attempts} times")
    finally:
        os.unlink(f.name)


def download(remote, local, timeout=120, attempts=1):
    """Quietly download one file from the VM: True if downloaded, None if it doesn't exist (yet), False if every
    attempt failed. Never touches the kernel.

    `colab download` occasionally hangs, so each attempt is killed after `timeout` seconds.
    """
    for _ in range(attempts):
        try:
            r = subprocess.run(["colab", "download", "-s", SESSION, remote, str(local)],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0:
            return True
        if "not found" in (r.stdout + r.stderr).lower():
            return None
    return False


def upload(local, remote, timeout=300, attempts=3):
    for _ in range(attempts):
        try:
            r = subprocess.run(["colab", "upload", "-s", SESSION, str(local), remote],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    sys.exit(f"upload of {local} failed {attempts} times")


def read_remote(remote, timeout=120):
    """Contents of a small text file on the VM, or None if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "f"
        return local.read_text() if download(remote, local, timeout) else None


# The supervisor cell. It runs in the kernel for as long as there is work, which keeps Colab from pruning the VM.
SUPERVISOR = r'''
import glob, hashlib, json, os, shlex, shutil, subprocess, sys, tarfile, time, traceback, zipfile
os.chdir(REMOTE)
os.makedirs("queue", exist_ok=True)
os.makedirs("runs", exist_ok=True)

def alive(pid):
    try:
        state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    if state == "Z":                              # finished but not reaped yet
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        return False
    return True

def running():
    return [os.path.dirname(f) for f in glob.glob("runs/*/pid")
            if not os.path.exists(os.path.join(os.path.dirname(f), "exit_code")) and alive(int(open(f).read()))]

def launch(run_dir, module, args):
    """Detached `python -m module args`, output to run_dir/train.log, exit status to run_dir/exit_code."""
    if os.path.exists(f"{run_dir}/pid") and alive(int(open(f"{run_dir}/pid").read())):
        return f"{run_dir} is already running, not started again"
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(f"{run_dir}/exit_code"):
        os.remove(f"{run_dir}/exit_code")
    cmd = [sys.executable, "-u", "-m", module, *args]
    script = f"{shlex.join(cmd)} >> {run_dir}/train.log 2>&1; echo $? > {run_dir}/exit_code"
    p = subprocess.Popen(["/bin/sh", "-c", script], stdin=subprocess.DEVNULL, start_new_session=True)
    open(f"{run_dir}/pid", "w").write(str(p.pid))
    return f"{run_dir} started (pid {p.pid}): {' '.join(cmd[2:])}"

def assemble(parts, sha256, dest):
    """Join uploaded chunks into dest, checking the checksum before replacing any existing file."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    sha = hashlib.sha256()
    with open(dest + ".tmp", "wb") as out:
        for p in parts:
            data = open(p, "rb").read(); sha.update(data); out.write(data); os.remove(p)
    assert sha.hexdigest() == sha256, "checksum mismatch after upload"
    os.replace(dest + ".tmp", dest)

def handle(job):
    kind = job["kind"]
    if kind == "sync":
        shutil.rmtree("src", ignore_errors=True)
        tarfile.open("queue/code.tgz").extractall(".", filter="data")
        os.remove("queue/code.tgz")
        return f"synced: {sorted(os.listdir('src'))}"
    if kind == "put":
        assemble(job["parts"], job["sha256"], job["dest"])
        return f"{job['dest']}: {os.path.getsize(job['dest']):,} bytes, checksum OK"
    if kind == "data":
        assemble(job["parts"], job["sha256"], "data.zip")
        zipfile.ZipFile("data.zip").extractall("data"); os.remove("data.zip")
        r = subprocess.run([sys.executable, "-c", "from src.data import load_split; "
                            "[print(s, load_split('data', s)[0].shape) for s in ('train', 'test')]"],
                           capture_output=True, text=True)
        return f"data.zip checksum OK\n{r.stdout}{r.stderr[-2000:]}"
    if kind == "train":
        run_dir = f"runs/{job['run']}"
        return launch(run_dir, "src.train", ["--out", run_dir, "--resume", *job["args"]])
    if kind == "eval":
        return launch(f"runs/{job['run']}", "src.evaluate", job["args"])
    raise ValueError(f"unknown job kind {kind!r}")

print("supervisor started", flush=True)
last_busy = time.time()
while True:
    for spec in sorted(glob.glob("queue/*.json")):
        try:
            job = json.load(open(spec))
        except ValueError:                        # still being uploaded
            continue
        os.remove(spec)
        try:
            result = handle(job)
        except Exception:
            result = "FAILED\n" + traceback.format_exc()
        print(result, flush=True)
        with open(spec[:-len(".json")] + ".done", "w") as f:
            f.write(result)
    with open("queue/heartbeat", "w") as f:
        f.write(str(time.time()))
    if running():
        last_busy = time.time()
    elif time.time() - last_busy > IDLE_EXIT:
        print("no training for a while, exiting so Colab can release the VM", flush=True)
        break
    time.sleep(5)
'''


def supervisor_alive():
    beat = read_remote(f"{REMOTE}/queue/heartbeat")
    return beat is not None and abs(time.time() - float(beat)) < 300  # the data job pauses it for a minute or two


def ensure_supervisor():
    """Start the supervisor cell (as a detached local `colab exec`) unless it is already running."""
    if supervisor_alive():
        return
    code = f"REMOTE = {REMOTE!r}\nIDLE_EXIT = {IDLE_EXIT}\n" + SUPERVISOR
    path = Path(tempfile.gettempdir()) / f"colab-supervisor-{SESSION}.py"
    path.write_text(code)
    SUPERVISOR_LOG.parent.mkdir(exist_ok=True)
    for attempt in range(3):
        print(f"[colab.py] starting the supervisor (log: {SUPERVISOR_LOG})", flush=True)
        with open(SUPERVISOR_LOG, "a") as log:
            proc = subprocess.Popen(["colab", "exec", "-s", SESSION, "--timeout", str(7 * 24 * 3600), "-f", str(path)],
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        for _ in range(12):
            time.sleep(10)
            if supervisor_alive():
                if sys.platform == "darwin":  # no idle sleep while connected (-s: no system sleep on AC power)
                    subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(proc.pid)], start_new_session=True)
                return
        proc.kill()  # the exec never connected; the kernel runs cells in order, so a retry can't double-start it
    sys.exit("the supervisor did not start; check the session with `colab status`")


def submit_job(kind, wait=600, **job):
    """Hand a job to the supervisor and print its result."""
    ensure_supervisor()
    name = f"{time.time_ns()}-{kind}"
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / f"{name}.json"
        spec.write_text(json.dumps({"kind": kind, **job}))
        upload(spec, f"{REMOTE}/queue/{name}.json")
    deadline = time.time() + wait
    while time.time() < deadline:
        result = read_remote(f"{REMOTE}/queue/{name}.done")
        if result is not None:
            print(result, flush=True)
            if result.startswith("FAILED"):
                sys.exit(f"{kind} job failed")
            return
        time.sleep(5)
    sys.exit(f"{kind} job not finished after {wait}s; check {SUPERVISOR_LOG}")


def cmd_up():
    colab("new", "-s", SESSION, "--gpu", GPU)
    remote_python(
        "import subprocess, os\n"
        f"os.makedirs({REMOTE!r}, exist_ok=True)\n"
        "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],"
        " capture_output=True, text=True).stdout, 'CPUs:', os.cpu_count())\n")
    ensure_supervisor()
    cmd_sync()
    cmd_data()
    colab("usage", check=False)


def put_file(local, kind, chunk=16 * 2**20, **job):
    """Upload a file in chunks (`colab upload` of 125 MB fails with HTTP 500); the supervisor reassembles it."""
    ensure_supervisor()  # it creates the queue folder the parts go into
    sha = hashlib.sha256()
    parts = []
    prefix = f"{time.time_ns()}-{Path(local).name}"
    with tempfile.TemporaryDirectory() as tmp, open(local, "rb") as f:
        while data := f.read(chunk):
            sha.update(data)
            part = Path(tmp) / f"{prefix}.part{len(parts):03d}"
            part.write_bytes(data)
            upload(part, f"{REMOTE}/queue/{part.name}")
            parts.append(f"queue/{part.name}")
            print(f"uploaded {part.name}", flush=True)
    submit_job(kind, wait=900, parts=parts, sha256=sha.hexdigest(), **job)


def cmd_data():
    put_file(ROOT / "data.zip", "data")


def cmd_put(local, dest=None):
    """Upload a local file to dest (relative to /content/proj), by default its own path relative to the repo."""
    path = Path(local).resolve()
    put_file(path, "put", dest=dest or path.relative_to(ROOT).as_posix())


def cmd_sync():
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "code.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(ROOT / "src", "src", filter=lambda t: None if "__pycache__" in t.name else t)
            tar.add(ROOT / "requirements.txt", "requirements.txt")
        ensure_supervisor()
        upload(archive, f"{REMOTE}/queue/code.tgz")
    submit_job("sync")


def cmd_start(run, *args):
    submit_job("train", run=run, args=list(args))


def cmd_train(run, *args):
    cmd_start(run, *args)
    cmd_follow(run)


def cmd_screen(epochs="40"):
    # One A100 is underused by a single 5M-parameter model, so the candidates share it.
    runs = [f"screen_{m}" for m in MODELS]
    for run, m in zip(runs, MODELS):
        cmd_start(run, "--model", m, "--epochs", str(epochs), "--num-workers", "4")
    cmd_follow(*runs)


def backup(run):
    """Download the run's resumable state to runs/RUN/last.pt, replacing the previous copy only if complete.

    True if backed up, None if the run has no last.pt yet (e.g. still waiting for its teachers), False on failure.
    """
    out = ROOT / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "last.pt.download"
    ok = download(f"{REMOTE}/runs/{run}/last.pt", tmp, timeout=600)
    if ok is None:
        return None
    try:  # torch.save writes a zip archive; testzip checks every member's CRC
        ok = ok and zipfile.ZipFile(tmp).testzip() is None
    except (zipfile.BadZipFile, OSError):
        ok = False
    if ok:
        os.replace(tmp, out / "last.pt")
    else:
        tmp.unlink(missing_ok=True)
    return ok


def cmd_follow(*runs, interval=20, fetch=True, backup_every=int(os.environ.get("COLAB_BACKUP_MIN", "10")) * 60):
    """Print new train.log lines of each run until it writes its exit_code, then fetch it.

    Every backup_every seconds (None: never; env COLAB_BACKUP_MIN sets the minutes), each run's last.pt is copied
    to runs/RUN/last.pt, so a lost VM costs at most that much training: `put` the copy on a new VM and `start` the
    run again. A WRN-28-10 teacher's last.pt is about 440 MB, so teachers are better backed up less often.
    """
    seen = {run: None for run in runs}
    pending = set(runs)
    last_backup = {run: time.time() for run in runs}
    with tempfile.TemporaryDirectory() as tmp:
        while pending:
            for run in sorted(pending):
                if backup_every and time.time() - last_backup[run] > backup_every:
                    last_backup[run] = time.time()
                    ok = backup(run)
                    if ok is not None:
                        print(f"[{run}] backed up last.pt" if ok else f"[{run}] last.pt backup failed", flush=True)
                log = Path(tmp) / f"{run}.log"
                if download(f"{REMOTE}/runs/{run}/train.log", log):
                    lines = log.read_text().splitlines()
                    start = max(0, len(lines) - 5) if seen[run] is None else seen[run]  # reattach: last 5 lines
                    for line in lines[start:]:
                        print(f"[{run}] {line}" if len(runs) > 1 else line, flush=True)
                    seen[run] = len(lines)
                code = read_remote(f"{REMOTE}/runs/{run}/exit_code")
                if code is not None:
                    print(f"[{run}] exited with code {code.strip()}", flush=True)
                    pending.discard(run)
                    if fetch:
                        cmd_fetch(run)
            if pending:
                time.sleep(interval)


def cmd_status(*runs):
    print(f"supervisor: {'alive' if supervisor_alive() else 'NOT running'}")
    for run in runs:
        text = read_remote(f"{REMOTE}/runs/{run}/history.json")
        if text is None:
            print(f"[{run}] no history yet")
            continue
        hist = json.loads(text)
        last = hist[-1]
        msg = (f"[{run}] epoch {last['epoch']} | train loss {last['train_loss']:.4f} acc {last['train_acc']:.2f} "
               f"| {last['time']:.1f}s/epoch")
        if last.get("val_acc") is not None:
            msg += f" | val acc {last['val_acc']:.2f} (best {max(h['val_acc'] for h in hist):.2f})"
        print(msg)


def cmd_eval(*args):
    """Run src.evaluate on the VM (checkpoint paths relative to /content/proj, e.g. runs/base_c/final.pt)."""
    run = f"eval_{time.strftime('%m%d-%H%M%S')}"
    submit_job("eval", run=run, args=list(args))
    cmd_follow(run, interval=10, fetch=False, backup_every=None)


def cmd_fetch(*runs):
    for run in runs:
        out = ROOT / "runs" / run
        out.mkdir(parents=True, exist_ok=True)
        for name in FETCH_FILES:
            ok = download(f"{REMOTE}/runs/{run}/{name}", out / name, timeout=600, attempts=3)
            print(f"[{run}] {name}: {'ok' if ok else 'not downloaded (missing on the VM, or timed out 3 times)'}",
                  flush=True)
        print(f"[{run}] saved to {out}")


def cmd_down():
    colab("stop", "-s", SESSION)


COMMANDS = {"up": cmd_up, "data": cmd_data, "put": cmd_put, "sync": cmd_sync, "start": cmd_start, "train": cmd_train,
            "screen": cmd_screen, "follow": cmd_follow, "status": cmd_status, "eval": cmd_eval,
            "fetch": cmd_fetch, "down": cmd_down}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    start = time.time()
    COMMANDS[sys.argv[1]](*sys.argv[2:])
    print(f"done in {time.time() - start:.0f}s")
