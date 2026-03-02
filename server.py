"""
FastAPI backend for TikTok video sorting UI.
Serves videos, provides predictions, and handles sorting into folders.

Run: python3 server.py
"""

import json
import re
import shutil
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

app = FastAPI()

CONFIG_FILE = Path(__file__).parent / "config.json"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
INDEX_HTML = Path(__file__).parent / "index.html"
SETUP_HTML = Path(__file__).parent / "setup.html"

# Default — overridden by config.json at startup
DATA_DIR = Path(__file__).parent / "data" / "Favorites" / "videos"

# Loaded at startup
predictions: dict[str, dict] = {}
retrain_status: dict = {"running": False, "last_result": None}

FILENAME_RE = re.compile(r"^\d+\.mp4$")
PROJECT_DIR = Path(__file__).parent


def load_config():
    global DATA_DIR
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            if "data_dir" in cfg:
                DATA_DIR = Path(cfg["data_dir"])
        except Exception:
            pass


def is_configured() -> bool:
    return DATA_DIR.exists() and DATA_DIR.is_dir()


class SortRequest(BaseModel):
    filename: str
    folder: str


class SetupRequest(BaseModel):
    data_dir: str


def get_folders():
    """Return list of {name, count} for all subfolders."""
    folders = []
    for item in sorted(DATA_DIR.iterdir()):
        if item.is_dir():
            count = len(list(item.glob("*.mp4")))
            folders.append({"name": item.name, "count": count})
    return folders


@app.on_event("startup")
def startup():
    load_config()
    _load_predictions()


def _load_predictions():
    pred_file = ARTIFACTS_DIR / "predictions.json"
    if pred_file.exists():
        data = json.loads(pred_file.read_text())
        for entry in data:
            predictions[entry["video"]] = entry


@app.get("/")
def serve_index():
    if not is_configured():
        return RedirectResponse("/setup")
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/setup")
def serve_setup():
    return FileResponse(SETUP_HTML, media_type="text/html")


@app.post("/api/setup")
def save_setup(req: SetupRequest):
    path = Path(req.data_dir.strip().strip('"'))

    if not path.exists():
        raise HTTPException(400, "Path does not exist")
    if not path.is_dir():
        raise HTTPException(400, "Path is not a directory")

    mp4_count = len(list(path.glob("*.mp4")))
    folders = {
        d.name: len(list((path / d.name).glob("*.mp4")))
        for d in sorted(path.iterdir())
        if d.is_dir()
    }

    CONFIG_FILE.write_text(json.dumps({"data_dir": str(path)}, indent=2))
    load_config()

    return {
        "success": True,
        "data_dir": str(path),
        "unsorted_videos": mp4_count,
        "folders": folders,
    }


@app.get("/api/config/status")
def config_status():
    return {
        "configured": is_configured(),
        "data_dir": str(DATA_DIR) if is_configured() else None,
    }


@app.get("/api/videos")
def list_videos():
    if not is_configured():
        raise HTTPException(503, "Not configured — visit /setup first")
    videos = []
    for f in sorted(DATA_DIR.glob("*.mp4")):
        if f.is_file():
            entry = predictions.get(f.name, {})
            videos.append({
                "filename": f.name,
                "predicted_folder": entry.get("predicted_folder", None),
                "confidence": entry.get("confidence", 0),
                "top_predictions": entry.get("top_predictions", []),
            })
    return {"videos": videos, "total": len(videos)}


@app.get("/api/folders")
def list_folders():
    if not is_configured():
        raise HTTPException(503, "Not configured — visit /setup first")
    return {"folders": get_folders()}


@app.post("/api/sort")
def sort_video(req: SortRequest):
    if not is_configured():
        raise HTTPException(503, "Not configured — visit /setup first")

    if not FILENAME_RE.match(req.filename):
        raise HTTPException(400, "Invalid filename")

    src = DATA_DIR / req.filename
    if not src.exists() or not src.is_file():
        raise HTTPException(404, "Video not found (may already be sorted)")

    dst_dir = DATA_DIR / req.folder
    if not dst_dir.exists() or not dst_dir.is_dir():
        raise HTTPException(400, f"Unknown folder: {req.folder}")

    if ".." in req.folder or "/" in req.folder:
        raise HTTPException(400, "Invalid folder name")

    dst = dst_dir / req.filename
    if dst.exists():
        raise HTTPException(409, "File already exists in target folder")

    shutil.move(str(src), str(dst))

    return {
        "success": True,
        "filename": req.filename,
        "folder": req.folder,
        "folders": get_folders(),
    }


def _run_retrain():
    """Run the full pipeline: extract features -> train -> predict."""
    retrain_status["running"] = True
    retrain_status["last_result"] = None
    try:
        for script in ["extract_features.py", "train.py", "predict.py"]:
            result = subprocess.run(
                ["python3", str(PROJECT_DIR / script)],
                capture_output=True, text=True, cwd=str(PROJECT_DIR),
                timeout=900,
            )
            if result.returncode != 0:
                retrain_status["last_result"] = f"Failed at {script}: {result.stderr[-500:]}"
                retrain_status["running"] = False
                return

        predictions.clear()
        _load_predictions()
        retrain_status["last_result"] = "success"
    except Exception as e:
        retrain_status["last_result"] = f"Error: {str(e)}"
    finally:
        retrain_status["running"] = False


@app.post("/api/retrain")
def retrain():
    if retrain_status["running"]:
        return {"status": "already_running"}
    thread = threading.Thread(target=_run_retrain, daemon=True)
    thread.start()
    return {"status": "started"}


@app.get("/api/retrain/status")
def retrain_progress():
    return {
        "running": retrain_status["running"],
        "last_result": retrain_status["last_result"],
    }


@app.get("/videos/{filename}")
def serve_video(filename: str):
    if not is_configured():
        raise HTTPException(503, "Not configured — visit /setup first")
    if not FILENAME_RE.match(filename):
        raise HTTPException(400, "Invalid filename")

    path = DATA_DIR / filename
    if not path.exists():
        for sub in DATA_DIR.iterdir():
            if sub.is_dir():
                candidate = sub / filename
                if candidate.exists():
                    path = candidate
                    break
        else:
            raise HTTPException(404, "Video not found")

    return FileResponse(path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
