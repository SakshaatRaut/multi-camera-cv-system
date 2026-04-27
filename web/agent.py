"""
Benchmark agent — runs the full multi-camera CV suite locally and uploads
the results to the coordinator dashboard.

Typical usage (on any participating laptop):

    python web/agent.py --server http://<coordinator-host>:8000 \\
                        --device-label "Sahil's M4 Air"

By default this runs `main.py --run-experiments all` with the
canonical 8-video / 30-second / YOLOv8n configuration, then bundles
`results/baseline/` and `experiments/results/` into one zip and POSTs
it to the coordinator.

You can pass `--mode quick` to only do the baseline run (about 30s)
for a low-effort first submission.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import urllib.request
    import urllib.error
except Exception:
    print("urllib is required (stdlib).")
    raise


# --------------------------------------------------------------------- #
# Hardware autodetect
# --------------------------------------------------------------------- #
def detect_hardware() -> dict:
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'cpu_count_logical': os.cpu_count(),
    }

    # psutil is optional but gives a much nicer picture
    try:
        import psutil  # type: ignore
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        try:
            info['cpu_freq_max_mhz'] = psutil.cpu_freq().max
        except Exception:
            pass
        try:
            info['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 2)
        except Exception:
            pass
    except Exception:
        pass

    # Accelerator info via torch if available
    try:
        import torch  # type: ignore
        info['torch'] = torch.__version__
        if torch.cuda.is_available():
            info['accelerator'] = 'cuda'
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            info['cuda_version'] = torch.version.cuda
        elif getattr(torch.backends, 'mps', None) and \
                torch.backends.mps.is_available():
            info['accelerator'] = 'mps'
            info['gpu_name'] = 'Apple Silicon GPU (MPS)'
        else:
            info['accelerator'] = 'cpu'
    except Exception:
        info['accelerator'] = 'unknown'

    return info


def detect_git_commit(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ['git', '-C', str(project_root), 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return ''


# --------------------------------------------------------------------- #
# Running main.py
# --------------------------------------------------------------------- #
def find_videos(project_root: Path, num: int = 8) -> list[str]:
    """Collect 8 video paths from the canonical locations, cycling if fewer."""
    candidates = []
    for d in [project_root / 'videos',
              project_root / 'data' / 'videos',
              project_root / 'sample_videos']:
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}:
                    candidates.append(str(p.resolve()))
    if not candidates:
        raise FileNotFoundError(
            f"No videos found under {project_root}/videos/. "
            "Place at least one .mp4 there before running the agent."
        )
    if len(candidates) >= num:
        return candidates[:num]
    # cycle to reach `num`
    out = []
    i = 0
    while len(out) < num:
        out.append(candidates[i % len(candidates)])
        i += 1
    return out


def run_main(project_root: Path, *, mode: str, device: str,
             duration: int, model: str,
             use_fp16: bool, use_gpu_preprocess: bool) -> int:
    """Invoke `python main.py ...`. Returns the subprocess exit code."""
    videos = find_videos(project_root, num=8)
    cmd = [
        sys.executable, 'main.py',
        '--video-sources', *videos,
        '--device', device,
        '--duration', str(duration),
        '--model', model,
    ]
    if use_fp16:
        cmd.append('--fp16')
    if use_gpu_preprocess:
        cmd.append('--gpu-preprocess')
    if mode == 'full':
        cmd += ['--run-experiments', 'all']
    elif mode == 'core':
        cmd += ['--run-experiments', 'core']
    # else: 'quick' -> no experiment suite

    print('\n' + '=' * 70)
    print('AGENT: running', ' '.join(cmd))
    print('=' * 70 + '\n')
    return subprocess.call(cmd, cwd=project_root)


# --------------------------------------------------------------------- #
# Bundle + upload
# --------------------------------------------------------------------- #
def build_bundle(project_root: Path, out_zip: Path, *,
                 device_label: str, hardware: dict, git_commit: str,
                 mode: str, notes: str) -> None:
    """Package baseline + experiments into a single zip."""
    baseline_dir = project_root / 'results' / 'baseline'
    results_dir = project_root / 'results'
    exp_dir = project_root / 'experiments' / 'results'

    manifest = {
        'device_label': device_label,
        'hardware': hardware,
        'git_commit': git_commit,
        'mode': mode,
        'notes': notes,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('manifest.json', json.dumps(manifest, indent=2))

        # Prefer baseline/ (preserved by --run-experiments); fall back to
        # results/ when the user did a quick-mode run.
        added_baseline = False
        if baseline_dir.exists() and any(baseline_dir.iterdir()):
            _add_dir_to_zip(zf, baseline_dir, arc_prefix='baseline')
            added_baseline = True
        elif results_dir.exists():
            for p in results_dir.iterdir():
                if p.is_file():
                    zf.write(p, arcname=f'baseline/{p.name}')
                    added_baseline = True

        if not added_baseline:
            raise RuntimeError('No baseline run outputs found in results/')

        # Experiment suite outputs (optional)
        if exp_dir.exists():
            _add_dir_to_zip(zf, exp_dir, arc_prefix='experiments')


def _add_dir_to_zip(zf: zipfile.ZipFile, src_dir: Path,
                    arc_prefix: str) -> None:
    for p in src_dir.rglob('*'):
        if p.is_file():
            rel = p.relative_to(src_dir)
            zf.write(p, arcname=f'{arc_prefix}/{rel.as_posix()}')


def upload_bundle(server_url: str, bundle_zip: Path, *,
                  device_label: str, hardware: dict, git_commit: str,
                  notes: str) -> dict:
    """POST multipart/form-data to /api/submissions."""
    # We build multipart by hand to avoid a `requests` dependency.
    boundary = f'----cowork-boundary-{int(time.time()*1000)}'
    lines: list[bytes] = []

    def field(name: str, value: str) -> None:
        lines.append(f'--{boundary}'.encode())
        lines.append(
            f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b'')
        lines.append(value.encode())

    field('device_label', device_label)
    field('hardware_json', json.dumps(hardware))
    field('git_commit', git_commit)
    field('notes', notes)

    # File part
    lines.append(f'--{boundary}'.encode())
    lines.append(
        ('Content-Disposition: form-data; name="bundle"; '
         'filename="bundle.zip"').encode())
    lines.append(b'Content-Type: application/zip')
    lines.append(b'')
    lines.append(bundle_zip.read_bytes())
    lines.append(f'--{boundary}--'.encode())
    lines.append(b'')
    body = b'\r\n'.join(lines)

    url = server_url.rstrip('/') + '/api/submissions'
    req = urllib.request.Request(
        url, data=body, method='POST',
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f'Server returned {exc.code}: {exc.read().decode(errors="ignore")}'
        ) from exc


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description='Benchmark agent')
    p.add_argument('--server', required=True,
                   help='Coordinator URL, e.g. http://100.x.y.z:8000')
    p.add_argument('--device-label', required=True,
                   help='Friendly name of this machine shown on the dashboard')
    p.add_argument('--mode', choices=['quick', 'core', 'full'], default='full',
                   help='quick = baseline only (~30s); '
                        'core = + E1-E4; full = + E1-E7 (20-40 min)')
    p.add_argument('--device', default='auto',
                   choices=['auto', 'cuda', 'mps', 'cpu'],
                   help='Passed to main.py')
    p.add_argument('--duration', type=int, default=30,
                   help='Baseline run duration (seconds)')
    p.add_argument('--model', default='yolov8n.pt',
                   help='Default YOLO variant for the baseline run')
    p.add_argument('--fp16', action='store_true',
                   help='Force FP16 baseline (CUDA only). '
                        'On CUDA this is auto-enabled unless --vanilla is set.')
    p.add_argument('--gpu-preprocess', action='store_true',
                   help='Force GPU preprocessing for the baseline. '
                        'On CUDA this is auto-enabled unless --vanilla is set.')
    p.add_argument('--vanilla', action='store_true',
                   help='Disable auto-optimization. Run a vanilla FP32+CPU-preproc '
                        'baseline regardless of accelerator. Useful for fair '
                        'cross-device comparison; the optimized numbers still '
                        'land in the E6/E7 experiment charts.')
    p.add_argument('--notes', default='',
                   help='Free-form notes shown next to the submission')
    p.add_argument('--skip-run', action='store_true',
                   help='Skip main.py and only upload existing results/ '
                        'and experiments/results/ folders')
    p.add_argument('--dry-run', action='store_true',
                   help='Build the bundle but do not upload')
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    hardware = detect_hardware()
    git_commit = detect_git_commit(project_root)

    # Auto-optimize the baseline on CUDA so the headline FPS bar reflects
    # realistic GPU throughput, not a deliberately-crippled FP32 number.
    # MPS doesn't gain from these flags (FP16 path is CUDA-only in our
    # detector), so we keep the existing behavior there. CPU obviously skips.
    accelerator = hardware.get('accelerator')
    auto_optimize = (accelerator == 'cuda') and not args.vanilla
    use_fp16 = args.fp16 or auto_optimize
    use_gpu_preprocess = args.gpu_preprocess or auto_optimize

    print('Agent starting with:')
    print(f'  server       : {args.server}')
    print(f'  device_label : {args.device_label}')
    print(f'  mode         : {args.mode}')
    print(f'  accelerator  : {accelerator}')
    print(f'  gpu_name     : {hardware.get("gpu_name", "-")}')
    print(f'  git_commit   : {git_commit or "(unknown)"}')
    print(f'  baseline fp16: {use_fp16}'
          + ('  (auto-enabled on CUDA)' if auto_optimize and not args.fp16
             else ''))
    print(f'  baseline gpu_preproc: {use_gpu_preprocess}'
          + ('  (auto-enabled on CUDA)' if auto_optimize and not args.gpu_preprocess
             else ''))

    if not args.skip_run:
        code = run_main(project_root, mode=args.mode,
                        device=args.device, duration=args.duration,
                        model=args.model,
                        use_fp16=use_fp16,
                        use_gpu_preprocess=use_gpu_preprocess)
        if code != 0:
            print(f'\nmain.py exited with code {code}.  Aborting upload.')
            sys.exit(code)

    bundle_path = project_root / 'web' / 'data' / 'bundle.zip'
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    build_bundle(project_root, bundle_path,
                 device_label=args.device_label,
                 hardware=hardware, git_commit=git_commit,
                 mode=args.mode, notes=args.notes)
    size_mb = bundle_path.stat().st_size / 1e6
    print(f'\nBuilt bundle: {bundle_path} ({size_mb:.1f} MB)')

    if args.dry_run:
        print('--dry-run set, not uploading.')
        return

    print(f'Uploading to {args.server} ...')
    resp = upload_bundle(args.server, bundle_path,
                         device_label=args.device_label,
                         hardware=hardware, git_commit=git_commit,
                         notes=args.notes)
    print('\nUpload successful:')
    print(json.dumps(resp, indent=2))
    base = args.server.rstrip('/')
    print(f'\nView on dashboard: {base}/')
    print(f'Submission JSON:   {base}{resp.get("url", "")}')


if __name__ == '__main__':
    main()
