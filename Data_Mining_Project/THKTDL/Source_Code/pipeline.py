import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ZIP = "https://github.com/KhoaDuong69/Data_Mining_Project/archive/refs/heads/main.zip"
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "_repo_cache"

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}


def in_cache(path, root):
    """True nếu path nằm trong _repo_cache (chỉ xét phần bên dưới root)."""
    try:
        return "_repo_cache" in path.relative_to(root).parts
    except ValueError:
        return False


def download_repo():
    """Xóa cache cũ, tải repo từ GitHub và trả về thư mục chứa Source_Code."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    print("Đang tải mã nguồn từ GitHub...", flush=True)
    try:
        with urllib.request.urlopen(REPO_ZIP, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"Không tải được từ GitHub: {e}")
        sys.exit(1)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(CACHE_DIR)
    return locate_root(CACHE_DIR)


def locate_root(search_dir):
    """Tìm thư mục cha của Source_Code bên trong search_dir."""
    for src in search_dir.rglob("Source_Code"):
        if src.is_dir() and not in_cache(src, search_dir):
            return src.parent
    print(f"Không thấy thư mục Source_Code trong {search_dir}")
    sys.exit(1)


def find_files(root):
    """Trả về (preprocess, notebook, app, danh_sách_thiếu)."""
    src = root / "Source_Code"
    preprocess = src / "TienXuLyDL.py"
    preprocess = preprocess if preprocess.exists() else None
    notebook = next(src.glob("Khai_th*.ipynb"), None)  # chịu được dấu cách thừa
    app = next((p for p in root.rglob("app.py") if not in_cache(p, root)), None)

    missing = [name for name, p in (
        ("Source_Code/TienXuLyDL.py", preprocess),
        ("Source_Code/Khai_th*.ipynb", notebook),
        ("app.py", app),
    ) if p is None]
    return preprocess, notebook, app, missing


def run_step(command, title, cwd, quiet=False):
    print(f"\n--- {title} ---", flush=True)
    if quiet:
        r = subprocess.run(command, cwd=str(cwd), env=ENV, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise subprocess.CalledProcessError(r.returncode, command)
    else:
        subprocess.run(command, cwd=str(cwd), env=ENV, check=True)
    print(f"Hoàn thành: {title}", flush=True)


def main():
    local = "--local" in sys.argv
    update = "--update" in sys.argv

    if local:
        root = locate_root(BASE_DIR)
        preprocess, notebook, app, missing = find_files(root)
    else:
        if update or not CACHE_DIR.exists():
            root = download_repo()
        else:
            root = locate_root(CACHE_DIR)
        preprocess, notebook, app, missing = find_files(root)
        if missing and not update:  # cache có thể đã cũ -> tải lại 1 lần
            print("Thiếu file trong bản cache, thử tải lại từ GitHub...")
            root = download_repo()
            preprocess, notebook, app, missing = find_files(root)

    if missing:
        print("Thiếu file trong repo:", ", ".join(missing))
        print("Kiểm tra: đã git push chưa, và file nằm ở nhánh main chưa.")
        sys.exit(1)

    print(f"Chế độ: {'local' if local else 'GitHub'}")
    print(f"  Thư mục gốc : {root}")
    print(f"  Tiền xử lý  : {preprocess.relative_to(root)}")
    print(f"  Notebook    : {notebook.relative_to(root)}")
    print(f"  App         : {app.relative_to(root)}")

    try:
        run_step([sys.executable, "-W", "ignore", str(preprocess)],
                 "Tiền xử lý dữ liệu", root, quiet=True)

        run_step([sys.executable, "-m", "jupyter", "nbconvert",
                  "--to", "notebook", "--execute", "--inplace",
                  "--ExecutePreprocessor.timeout=600", str(notebook)],
                 "Khai thác tập phổ biến", root, quiet=True)

        run_step([sys.executable, "-m", "streamlit", "run", str(app)],
                 "Khởi chạy ứng dụng", root)

    except KeyboardInterrupt:
        print("\nĐã dừng ứng dụng.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline dừng do bước chạy bị lỗi (mã {e.returncode}).")
        sys.exit(e.returncode or 1)


if __name__ == "__main__":
    main()