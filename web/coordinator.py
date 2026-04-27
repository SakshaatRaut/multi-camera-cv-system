"""
Async benchmark coordinator for the Multi-Camera CV leaderboard.

Accepts submissions from agents running on heterogeneous hardware
(CUDA GPUs + Apple Silicon) and serves a dashboard that overlays
their results for side-by-side comparison.

Endpoints
---------
GET  /                           -> dashboard.html
GET  /api/submissions            -> JSON array of all submissions (summary)
GET  /api/submissions/{id}       -> full submission detail
GET  /api/submissions/{id}/plot/{name}.png
                                 -> serves a plot file for the submission
POST /api/submissions            -> accept a new submission (multipart/form-data)
DELETE /api/submissions/{id}     -> remove a submission (admin-ish, no auth)

Storage
-------
SQLite db at web/data/runs.db (metadata + JSON blobs)
Filesystem at web/data/submissions/{id}/ (preserved artifacts and plots)

Run
---
    uvicorn web.coordinator:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


# --------------------------------------------------------------------- #
# Paths + DB setup
# --------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data'
SUBMISSIONS_DIR = DATA_DIR / 'submissions'
DB_PATH = DATA_DIR / 'runs.db'
DASHBOARD_HTML = HERE / 'dashboard.html'

DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id            TEXT PRIMARY KEY,
                device_label  TEXT NOT NULL,
                submitted_at  TEXT NOT NULL,
                hardware      TEXT,       -- JSON
                git_commit    TEXT,
                baseline      TEXT,       -- JSON: run_summary.json
                experiments   TEXT,       -- JSON: experiment_metadata.json
                per_run       TEXT,       -- JSON: map of experiment -> per-run summaries
                notes         TEXT
            )
        """)
        c.commit()


_init_db()


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
    """Return the lightweight summary used by the leaderboard list view."""
    hardware = json.loads(row['hardware'] or '{}')
    baseline = json.loads(row['baseline'] or '{}')
    experiments = json.loads(row['experiments'] or '{}')

    pipeline = (baseline.get('pipeline') or {}) if baseline else {}
    config = (baseline.get('config') or {}) if baseline else {}

    return {
        'id': row['id'],
        'device_label': row['device_label'],
        'submitted_at': row['submitted_at'],
        'git_commit': row['git_commit'],
        'hardware': hardware,
        'has_experiments': bool(experiments),
        'headline': {
            'overall_fps': pipeline.get('overall_fps'),
            'avg_latency_ms': pipeline.get('avg_latency_ms'),
            'total_frames': pipeline.get('total_frames'),
            'num_cameras': config.get('num_cameras'),
            'device_mode': config.get('device'),
            'backend': config.get('backend'),
            'fp16': config.get('use_fp16'),
        },
    }


def _submission_dir(sub_id: str) -> Path:
    return SUBMISSIONS_DIR / sub_id


def _read_json_safe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# --------------------------------------------------------------------- #
# FastAPI
# --------------------------------------------------------------------- #
app = FastAPI(title='Multi-Camera CV Benchmark Coordinator')


@app.get('/', response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    if not DASHBOARD_HTML.exists():
        return HTMLResponse(
            '<h1>Dashboard not built yet</h1>'
            '<p>Place <code>dashboard.html</code> next to <code>coordinator.py</code>.</p>',
            status_code=200,
        )
    return HTMLResponse(DASHBOARD_HTML.read_text())


@app.get('/api/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'time': datetime.now(timezone.utc).isoformat()}


@app.get('/api/submissions')
def list_submissions() -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            'SELECT * FROM submissions ORDER BY submitted_at DESC'
        ).fetchall()
    return [_row_to_summary(r) for r in rows]


@app.get('/api/submissions/{sub_id}')
def get_submission(sub_id: str) -> dict[str, Any]:
    with _conn() as c:
        row = c.execute(
            'SELECT * FROM submissions WHERE id = ?', (sub_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail='submission not found')

    detail = _row_to_summary(row)
    detail['baseline'] = json.loads(row['baseline'] or '{}')
    detail['experiments'] = json.loads(row['experiments'] or '{}')
    detail['per_run'] = json.loads(row['per_run'] or '{}')
    detail['notes'] = row['notes']

    # List available plot files
    plots_dir = _submission_dir(sub_id) / 'plots'
    plots = []
    if plots_dir.exists():
        plots = sorted(p.name for p in plots_dir.iterdir()
                       if p.suffix.lower() == '.png')
    detail['plots'] = plots
    return detail


@app.get('/api/submissions/{sub_id}/plot/{plot_name}')
def get_plot(sub_id: str, plot_name: str) -> FileResponse:
    # Guard against path traversal
    if '/' in plot_name or '\\' in plot_name or '..' in plot_name:
        raise HTTPException(status_code=400, detail='invalid plot name')
    p = _submission_dir(sub_id) / 'plots' / plot_name
    if not p.exists():
        raise HTTPException(status_code=404, detail='plot not found')
    return FileResponse(p, media_type='image/png')


@app.delete('/api/submissions/{sub_id}')
def delete_submission(sub_id: str) -> dict[str, Any]:
    with _conn() as c:
        cur = c.execute('DELETE FROM submissions WHERE id = ?', (sub_id,))
        c.commit()
        deleted = cur.rowcount
    # Remove the filesystem payload too
    d = _submission_dir(sub_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return {'deleted': deleted, 'id': sub_id}


@app.post('/api/submissions')
async def create_submission(
    device_label: str = Form(...),
    hardware_json: str = Form('{}'),
    git_commit: str = Form(''),
    notes: str = Form(''),
    bundle: UploadFile = File(...),
) -> dict[str, Any]:
    """Ingest a bundle produced by the agent.

    The bundle is a zip with this layout (relative paths):
        manifest.json              (optional: device_label + hardware + git)
        baseline/run_summary.json
        baseline/*.png             (histograms, etc.)
        baseline/*.txt
        experiments/run_summary.json-equivalents as <prefix>_summary.json
        experiments/experiment_metadata.json
        experiments/*.png          (e2 curve, e3 amdahl, e5/e6/e7 plots)
    """
    sub_id = uuid.uuid4().hex[:12]
    sub_dir = _submission_dir(sub_id)
    sub_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = sub_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Save the bundle to disk then extract
    bundle_path = sub_dir / 'bundle.zip'
    content = await bundle.read()
    bundle_path.write_bytes(content)

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(sub_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail='bundle is not a valid zip')

    # Parse JSON payloads
    baseline_path = sub_dir / 'baseline' / 'run_summary.json'
    baseline = _read_json_safe(baseline_path)

    exp_meta_path = sub_dir / 'experiments' / 'experiment_metadata.json'
    experiments = _read_json_safe(exp_meta_path)

    # Collect per-experiment per-run summaries (e1_gpu_summary.json etc.)
    per_run: dict[str, Any] = {}
    exp_dir = sub_dir / 'experiments'
    if exp_dir.exists():
        for p in exp_dir.glob('*_summary.json'):
            try:
                per_run[p.stem] = json.loads(p.read_text())
            except Exception:
                pass

    # Copy all PNGs into plots/ with namespaced filenames for the dashboard
    for src_dir, prefix in [
        (sub_dir / 'baseline', 'baseline'),
        (sub_dir / 'experiments', 'exp'),
    ]:
        if not src_dir.exists():
            continue
        for p in src_dir.glob('*.png'):
            dst = plots_dir / f'{prefix}__{p.name}'
            try:
                shutil.copy2(p, dst)
            except Exception:
                pass

    # Allow agent to override identity via manifest.json in the zip
    manifest = _read_json_safe(sub_dir / 'manifest.json')
    effective_label = manifest.get('device_label') or device_label
    effective_commit = manifest.get('git_commit') or git_commit
    try:
        hardware = json.loads(hardware_json) if hardware_json else {}
    except Exception:
        hardware = {}
    if manifest.get('hardware'):
        hardware = {**hardware, **manifest['hardware']}

    submitted_at = datetime.now(timezone.utc).isoformat()

    with _conn() as c:
        c.execute(
            """
            INSERT INTO submissions
              (id, device_label, submitted_at, hardware, git_commit,
               baseline, experiments, per_run, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sub_id,
                effective_label,
                submitted_at,
                json.dumps(hardware),
                effective_commit,
                json.dumps(baseline),
                json.dumps(experiments),
                json.dumps(per_run),
                notes,
            ),
        )
        c.commit()

    return JSONResponse(
        {
            'id': sub_id,
            'submitted_at': submitted_at,
            'device_label': effective_label,
            'has_baseline': bool(baseline),
            'has_experiments': bool(experiments),
            'num_plots': len(list(plots_dir.iterdir())),
            'url': f'/api/submissions/{sub_id}',
        },
        status_code=201,
    )


if __name__ == '__main__':
    # Dev server. In production use: uvicorn web.coordinator:app --host 0.0.0.0
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
