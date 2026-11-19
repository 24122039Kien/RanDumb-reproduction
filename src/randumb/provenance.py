"""Dấu vết source cho các run E1/E8; không đọc credential hoặc dữ liệu riêng."""
from hashlib import sha256
from pathlib import Path
import subprocess


def code_version() -> dict:
    """Trả về commit, trạng thái chỉnh sửa và SHA256 của source ảnh hưởng E1/E8."""
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src" / "randumb").glob("*.py"))
    files += [root / "scripts" / name for name in ["run_experiment.py", "run_sprint3.py"]]
    digest = sha256()
    for path in files:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": digest.hexdigest()}
