# Multi-Object Tracking — YOLOv8 & OpenCV

Track objects across video frames with **YOLOv8** detection and **ByteTrack / BoT-SORT** tracking. Every object gets a stable ID, and the script counts **unique** objects per class — ready for people counting, vehicle counting and video analytics tasks.

## Features
- YOLOv8 detection + persistent tracking IDs (`model.track` with `persist=True`)
- Choice of tracker: ByteTrack (default) or BoT-SORT
- Live overlay of unique counts per class, final count summary at the end
- Filter by class name (`--classes person car`)
- Works on video files or webcam, optional saving of the annotated video

## Tech Stack
Python • YOLOv8 (Ultralytics) • OpenCV • ByteTrack / BoT-SORT

## How to Run
```bash
pip install -r requirements.txt
python object_tracker.py --source video.mp4 --show
python object_tracker.py --source 0 --show                        # webcam
python object_tracker.py --source video.mp4 --output output.mp4   # save result
python object_tracker.py --source video.mp4 --classes person car --tracker botsort.yaml
```

## Project Structure
```
yolov8-object-tracking/
├── object_tracker.py   # main script
├── requirements.txt
└── README.md
```

## Author
Abdul Hanan — Computer Vision & AI Developer
GitHub: https://github.com/alishan24456-glitch
LinkedIn: https://www.linkedin.com/in/ali-raza-computervision/
