"""Download a subset of the Coswara dataset (github.com/iiscleap/Coswara-Data).

Audio is extracted to  data/coswara/<date>/<participant_id>/*.wav
Metadata is saved to   data/metadata.csv  (the repo's combined_data.csv)

The full dataset is ~13 GB, so by default only a few date folders (~0.7 GB)
with a mix of COVID-positive and healthy participants are fetched.
Use --folders to choose others, or --all for everything.

Usage:
    python src/download_data.py                 # default subset
    python src/download_data.py --folders 20210630 20210714
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import requests

REPO = "iiscleap/Coswara-Data"
BRANCH = "master"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
API = f"https://api.github.com/repos/{REPO}/"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "coswara")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")

# Picked for a mix of labels at modest size (positives + healthy, ~0.7 GB).
DEFAULT_FOLDERS = ["20200604", "20200820", "20200919", "20200930", "20210714"]


def _get(path, stream=False):
    """Fetch a repo file: raw.githubusercontent first, GitHub API (raw media type) as fallback."""
    errors = []
    for url, headers in (
        (RAW + path, {}),
        (API + "contents/" + path + f"?ref={BRANCH}", {"Accept": "application/vnd.github.raw"}),
    ):
        try:
            r = requests.get(url, headers=headers, stream=stream, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            errors.append(f"{url.split('/')[2]}: {e}")
    raise RuntimeError("; ".join(errors))


def list_chunks():
    """Map date folder -> sorted list of tar.gz chunk paths, using one git-tree API call."""
    r = requests.get(API + f"git/trees/{BRANCH}?recursive=1", timeout=60)
    r.raise_for_status()
    chunks = {}
    for e in r.json()["tree"]:
        if e["type"] == "blob" and ".tar.gz." in e["path"]:
            chunks.setdefault(e["path"].split("/")[0], []).append(e["path"])
    return {k: sorted(v) for k, v in chunks.items()}


def _safe_extract(tar, dest):
    """Extract only regular files/dirs that stay inside dest (no links, no path traversal)."""
    dest = os.path.realpath(dest)
    for m in tar:  # sequential iteration also works for stream ("r|gz") archives
        target = os.path.realpath(os.path.join(dest, m.name))
        if not target.startswith(dest + os.sep) or not (m.isfile() or m.isdir()):
            continue
        tar.extract(m, dest)


def download_folder(folder, chunk_paths):
    if os.path.isdir(os.path.join(AUDIO_DIR, folder)):
        print(f"  {folder}: already extracted, skipping")
        return
    # Stream chunks into a temp file on disk (some folders are >1 GB), extract, then delete it.
    with tempfile.NamedTemporaryFile(dir=DATA_DIR, suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
        for p in chunk_paths:
            print(f"  {folder}: downloading {os.path.basename(p)} ...", flush=True)
            r = _get(p, stream=True)
            for block in r.iter_content(1 << 20):
                tmp.write(block)
    try:
        with tarfile.open(tmp_path, mode="r:gz") as tar:
            _safe_extract(tar, AUDIO_DIR)
    finally:
        os.unlink(tmp_path)
    print(f"  {folder}: extracted")


class _ChainedFiles:
    """Read several files back to back as one stream (the split .tar.gz.aa/.ab/... chunks)."""

    def __init__(self, paths):
        self._paths, self._f = list(paths), None

    def read(self, n=-1):
        while self._paths or self._f:
            if self._f is None:
                self._f = open(self._paths.pop(0), "rb")
            data = self._f.read(n)
            if data:
                return data
            self._f.close()
            self._f = None
        return b""


def git_list_folders():
    with tempfile.TemporaryDirectory(dir=DATA_DIR) as tmp:
        _git_clone(tmp, [])
        shutil.copy(os.path.join(tmp, "combined_data.csv"), META_PATH)
        names = subprocess.run(["git", "-C", tmp, "ls-tree", "--name-only", "HEAD"],
                               check=True, capture_output=True, text=True).stdout.split()
    return sorted(n for n in names if n.isdigit() and len(n) == 8)


def _git_clone(dest, folders):
    """Shallow, blob-less clone that materialises only the given top-level folders."""
    def run(*a):
        # HTTP/1.1: large (~1 GB) blob fetches over HTTP/2 get stream resets on flaky links.
        r = subprocess.run(["git", "-c", "http.version=HTTP/1.1", *a], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"git {a[0] if a[0] != '-C' else a[2]} failed: {r.stderr.strip()[-300:]}")

    run("clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
        f"https://github.com/{REPO}.git", dest)
    run("-C", dest, "sparse-checkout", "init", "--cone")
    run("-C", dest, "sparse-checkout", "set", *(folders or ["technical_validation"]))
    run("-C", dest, "checkout", "-q", BRANCH)


def download_folder_git(folder, retries=3):
    """Clone one folder via git (no API rate limit), extract it, delete the clone.

    Extraction goes to a staging dir that is moved into place only when complete, so an
    interrupted run never leaves a half-filled folder that a rerun would skip.
    Returns True on success, False after `retries` failed attempts.
    """
    final = os.path.join(AUDIO_DIR, folder)
    if os.path.isdir(final):
        print(f"  {folder}: already extracted, skipping")
        return True
    for attempt in range(1, retries + 1):
        tmp = tempfile.mkdtemp(dir=DATA_DIR, prefix="clone_")
        stage = tempfile.mkdtemp(dir=DATA_DIR, prefix="stage_")
        try:
            print(f"  {folder}: cloning (attempt {attempt}/{retries}) ...", flush=True)
            _git_clone(tmp, [folder])
            shutil.copy(os.path.join(tmp, "combined_data.csv"), META_PATH)  # top-level files come with cone mode
            chunks = sorted(glob.glob(os.path.join(tmp, folder, f"{folder}.tar.gz.*")))
            if not chunks:
                raise RuntimeError(f"no archive chunks found for {folder}")
            with tarfile.open(fileobj=_ChainedFiles(chunks), mode="r|gz") as tar:
                _safe_extract(tar, stage)
            extracted = os.path.join(stage, folder)
            if not os.path.isdir(extracted):
                raise RuntimeError(f"archive for {folder} did not contain a {folder}/ folder")
            os.rename(extracted, final)
            print(f"  {folder}: extracted", flush=True)
            return True
        except Exception as e:
            print(f"  {folder}: attempt {attempt} failed: {e}", flush=True)
            if attempt < retries:
                time.sleep(30 * attempt)  # back off so a brief network outage doesn't burn every retry
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(stage, ignore_errors=True)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS)
    ap.add_argument("--all", action="store_true", help="download every date folder (~13 GB)")
    ap.add_argument("--method", choices=["http", "git"], default="http",
                    help="git: per-folder sparse clones, avoids the 60 req/h API limit (use with --all)")
    args = ap.parse_args()
    os.makedirs(AUDIO_DIR, exist_ok=True)

    try:
        if args.method == "git":
            folders = git_list_folders() if args.all else args.folders
            failed = [f for f in folders if not download_folder_git(f)]
            if failed:
                print(f"\nWARNING: {len(failed)} folder(s) failed; rerun the same command to retry: {failed}")
        else:
            print("Fetching metadata ...")
            with open(META_PATH, "wb") as f:
                f.write(_get("combined_data.csv").content)
            chunks = list_chunks()
            folders = sorted(chunks) if args.all else args.folders
            missing = [f for f in folders if f not in chunks]
            if missing:
                print(f"WARNING: unknown folders skipped: {missing}")
            for f in folders:
                if f in chunks:
                    download_folder(f, chunks[f])
    except Exception as e:  # network blocked, rate-limited, etc.
        print("\nERROR: could not download Coswara data.\n"
              f"  Reason: {e}\n"
              "  Fix: check your internet connection / GitHub API rate limit (60 req/h unauthenticated),\n"
              f"  or clone manually: git clone https://github.com/{REPO}  and extract the\n"
              "  <date>/*.tar.gz.* chunks into data/coswara/, copying combined_data.csv to data/metadata.csv.")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
