# This comparison/analysis between devices works when multiple devices are connected to the same network and run accordingly

# For Host

# python main.py --video-sources videos/cam{1..8}.mp4 --device auto \--duration 30 --run-experiments all

# terminal 1 — run the coordinator

pip install -r web/requirements.txt
uvicorn web.coordinator:app --host 0.0.0.0 --port 8000

# terminal 2 — submit your existing results without re-running the 40-min suite

python web/agent.py --server http://localhost:8000 --device-label "Your GPU Name" --skip-run

# open http://10.0.0.41:8000/ in a browser


# For Mac Users

python main.py --video-sources videos/cam{1..8}.mp4 --device auto \--duration 30 --run-experiments all

# terminal 1 — run the coordinator

pip install -r web/requirements.txt
curl http://10.0.0.41:8000/api/health
# Then type A

# terminal 2 — submit your existing results without re-running the 40-min suite

python web/agent.py --server http://10.0.0.41:8000 --device-label "YOUR GPU NAME" --skip-run

# open http://10.0.0.41:8000/ in a browser



# For Windows Users

python main.py --video-sources videos/cam{1..8}.mp4 --device auto --duration 30 --run-experiments all

# terminal 1 — run the coordinator

pip install -r web/requirements.txt
curl http://10.0.0.41:8000/api/health
# Then Type A

# terminal 2 — submit your existing results without re-running the 40-min suite

python web/agent.py --server http://10.0.0.41:8000 --device-label "YOUR GPU NAME" --skip-run

# open http://10.0.0.41:8000/ in a browser
