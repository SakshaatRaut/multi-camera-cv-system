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
        # Always read as UTF-8 — submissions come from Windows/Mac/Linux
        # agents, all of which write UTF-8 by our convention.
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _synthesize_experiments_from_per_run(per_run: dict[str, Any]) -> dict[str, Any]:
    """Derive the experiments.E1-E7 structure from per_run summaries.

    `per_run` is keyed by the file stem of each per-experiment summary
    (e.g. 'e6_onnxruntime_summary'). The dashboard renders charts from
    `experiments.E1.runs`, `experiments.E2.runs`, etc. — the same shape
    that experiment_metadata.json produces.

    We synthesize this structure here so that:
      * Re-running a single experiment (e.g. `--run-experiments e6`)
        properly updates the dashboard, even though that path doesn't
        regenerate experiment_metadata.json.
      * Existing submissions in the DB pick up the latest per_run data
        without needing a re-upload.
    """
    if not per_run:
        return {}

    def _fps(summary):
        return ((summary or {}).get('pipeline') or {}).get('overall_fps')

    def _lat(summary):
        return ((summary or {}).get('pipeline') or {}).get('avg_latency_ms')

    out: dict[str, Any] = {}

    # E1: gpu vs cpu
    e1_runs = []
    for label in ('gpu', 'cpu'):
        s = per_run.get(f'e1_{label}_summary')
        if s:
            e1_runs.append({
                'label': label,
                'overall_fps': _fps(s),
                'avg_latency_ms': _lat(s),
            })
    if e1_runs:
        out['E1'] = {'runs': e1_runs}

    # E2: batch sizes
    e2_runs = []
    for bs in (1, 2, 4, 8, 16):
        s = per_run.get(f'e2_batch_{bs}_summary')
        if s:
            e2_runs.append({
                'batch_size': bs,
                'fps': _fps(s),
                'latency_ms': _lat(s),
            })
    if e2_runs:
        out['E2'] = {'runs': e2_runs}

    # E3: stream counts
    e3_runs = []
    for n in (1, 2, 4, 8):
        s = per_run.get(f'e3_streams_{n}_summary')
        if s:
            e3_runs.append({
                'streams': n,
                'fps': _fps(s),
                'latency_ms': _lat(s),
            })
    if e3_runs:
        out['E3'] = {'runs': e3_runs}

    # E5: model variants
    e5_runs = []
    for variant in ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'):
        s = per_run.get(f'e5_{variant}_summary')
        if s:
            detections = ((s or {}).get('detector') or {}).get('total_detections')
            e5_runs.append({
                'model': f'{variant}.pt',
                'fps': _fps(s),
                'latency_ms': _lat(s),
                'total_detections': detections,
            })
    if e5_runs:
        out['E5'] = {'runs': e5_runs}

    # E6: inference engines
    e6_runs = []
    for label in ('pytorch_fp32', 'pytorch_fp16', 'onnxruntime'):
        s = per_run.get(f'e6_{label}_summary')
        if s:
            e6_runs.append({
                'scenario': label,
                'fps': _fps(s),
                'latency_ms': _lat(s),
            })
    if e6_runs:
        out['E6'] = {'runs': e6_runs}

    # E7: GPU preprocessing scenarios
    e7_runs = []
    for label in ('cpu_preproc', 'gpu_preproc', 'gpu_preproc_fp16'):
        s = per_run.get(f'e7_{label}_summary')
        if s:
            e7_runs.append({
                'scenario': label,
                'fps': _fps(s),
                'latency_ms': _lat(s),
            })
    if e7_runs:
        out['E7'] = {'runs': e7_runs}

    return out


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
    return HTMLResponse(DASHBOARD_HTML.read_text(encoding='utf-8'))


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
    stored_experiments = json.loads(row['experiments'] or '{}')
    detail['per_run'] = json.loads(row['per_run'] or '{}')
    detail['notes'] = row['notes']

    # Always synthesize the experiments.E1-E7 chart structure from per_run.
    # This is authoritative (per_run reflects the latest individual runs),
    # whereas the stored `experiments` blob only refreshes when the full
    # suite runs to completion. We merge synthesized data over the stored
    # blob so that any extra metadata (e.g. e3 amdahl block) is preserved.
    synthesized = _synthesize_experiments_from_per_run(detail['per_run'])
    merged_experiments = dict(stored_experiments)
    inner = dict((stored_experiments or {}).get('experiments') or {})
    for key, value in synthesized.items():
        inner[key] = value
    if inner:
        merged_experiments['experiments'] = inner
    detail['experiments'] = merged_experiments

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
                per_run[p.stem] = json.loads(p.read_text(encoding='utf-8'))
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
