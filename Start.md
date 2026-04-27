# python main.py --video-sources videos/cam{1..8}.mp4 --device auto \--duration 30 --run-experiments all


# terminal 1 — run the coordinator
pip install -r web/requirements.txt
uvicorn web.coordinator:app --host 0.0.0.0 --port 8000

# terminal 2 — submit your existing results without re-running the 40-min suite
python web/agent.py --server http://localhost:8000 \
    --device-label "Sahil's M4 Air" --skip-run

# open http://localhost:8000/ in a browser