# This comparison/analysis between devices works when multiple devices are connected to the same network and run accordingly

## For Host
uvicorn web.coordinator:app --host 0.0.0.0 --port 8000

## For Mac Users
1. pip install -r requirements.txt

2. python main.py --video-sources videos/cam{1..8}.mp4 --device auto \--duration 30 --run-experiments all

3. curl http://10.0.0.41:8000/api/health

4. python web/agent.py --server http://10.0.0.41:8000 --device-label "YOUR GPU NAME" --skip-run


## For Windows Users
1. pip install -r requirements.txt

2. python main.py --video-sources videos/cam1.mp4 videos/cam2.mp4 videos/cam3.mp4 videos/cam4.mp4 videos/cam5.mp4 videos/cam6.mp4 videos/cam7.mp4 videos/cam8.mp4 --device auto --duration 30 --run-experiments all

3. curl http://10.0.0.41:8000/api/health

4. python web/agent.py --server http://10.0.0.41:8000 --device-label "YOUR GPU NAME" --skip-run