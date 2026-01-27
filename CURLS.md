# API Curl Tests (localhost)

These examples target a FastAPI server on `http://localhost:8000`. Replace
`{room_id}` and `{camera_id}` with your values.

## Health
```bash
curl -s http://localhost:8000/health | jq
```

## Rooms
```bash
curl -s http://localhost:8000/rooms | jq

curl -s -X POST http://localhost:8000/rooms \
  -H "Content-Type: application/json" \
  -d '{"room_id":"room1"}' | jq

curl -s -X DELETE http://localhost:8000/rooms/{room_id} | jq
```

## Cameras
```bash
curl -s -X POST http://localhost:8000/rooms/{room_id}/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","url":"rtsp://4.182.65.195:8554/camera2","role":"other"}' | jq

curl -s -X DELETE http://localhost:8000/rooms/{room_id}/cameras/{camera_id} | jq
```

## Activity (Motion Gate)
```bash
curl -s http://localhost:8000/activity | jq

curl -s http://localhost:8000/rooms/{room_id}/activity | jq

curl -s http://localhost:8000/rooms/{room_id}/cameras/{camera_id}/activity | jq
```

## Room Health
```bash
curl -s http://localhost:8000/rooms/{room_id}/health | jq
```

## Snapshot (JPEG)
```bash
curl -s http://localhost:8000/rooms/{room_id}/cameras/{camera_id}/snapshot \
  -o /tmp/{room_id}_{camera_id}_snapshot.jpg
```

## Perception Events
```bash
curl -s http://localhost:8000/perception/events | jq

curl -s "http://localhost:8000/perception/events?limit=50" | jq

curl -s "http://localhost:8000/perception/events?since=1700000000" | jq

curl -s "http://localhost:8000/perception/events?room_id={room_id}&camera_id={camera_id}" | jq
```

## Inference Outputs
```bash
curl -s http://localhost:8000/inference/outputs | jq

curl -s "http://localhost:8000/inference/outputs?limit=50" | jq

curl -s "http://localhost:8000/inference/outputs?since=1700000000" | jq
```

## Attendance
```bash
curl -s http://localhost:8000/attendance/today | jq

curl -s http://localhost:8000/attendance/2026-01-27 | jq
```

## Dashboard Summary
```bash
curl -s http://localhost:8000/dashboard/summary | jq

curl -s "http://localhost:8000/dashboard/summary?window_seconds=600" | jq
```
