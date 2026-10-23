"""
Multi-Object Tracking in Videos using YOLOv8 (ByteTrack / BoT-SORT) & OpenCV.

Detects objects with YOLOv8 and tracks them across frames, giving every
object a stable ID. Also counts unique objects per class — useful for
people counting, vehicle counting and similar video analytics.

Usage:
    python object_tracker.py --source video.mp4 --show
    python object_tracker.py --source 0 --show                  # webcam
    python object_tracker.py --source video.mp4 --output out.mp4
    python object_tracker.py --source video.mp4 --classes person car

Author: Abdul Hanan
"""

import argparse
import sys
from collections import defaultdict

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-object tracking in videos using YOLOv8."
    )
    parser.add_argument("--source", default="0",
                        help="Video file path, or '0' for the webcam (default: 0).")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 model weights (default: yolov8n.pt).")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        choices=["bytetrack.yaml", "botsort.yaml"],
                        help="Tracker config (default: bytetrack.yaml).")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4).")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class names to track, e.g. --classes person car. "
                             "Default: track every class the model knows.")
    parser.add_argument("--output", default=None,
                        help="Optional path to save the annotated video.")
    parser.add_argument("--show", action="store_true",
                        help="Show the annotated video while processing.")
    return parser.parse_args()


def resolve_class_ids(model: YOLO, requested: list[str] | None) -> list[int] | None:
    if requested is None:
        return None
    names: dict[int, str] = model.names
    wanted = {name.lower() for name in requested}
    matched = [cid for cid, name in names.items() if name.lower() in wanted]
    missing = wanted - {names[cid].lower() for cid in matched}
    if missing:
        print(f"Warning: class(es) {sorted(missing)} are not in this model. "
              f"The model knows: {sorted(names.values())}.", file=sys.stderr)
    if not matched:
        raise SystemExit("None of the requested classes exist in this model.")
    return matched


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    class_ids = resolve_class_ids(model, args.classes)

    source: int | str = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    # Unique track IDs seen per class -> unique object counts.
    seen_ids: dict[str, set[int]] = defaultdict(set)

    print("Tracking... press 'q' to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            classes=class_ids,
            verbose=False,
        )[0]

        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id in zip(result.boxes, result.boxes.id):
                class_name = model.names[int(box.cls[0])]
                seen_ids[class_name].add(int(track_id))

        annotated = result.plot()
        y = 30
        for class_name, ids in sorted(seen_ids.items()):
            cv2.putText(annotated, f"{class_name}: {len(ids)} unique",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 28

        if writer is not None:
            writer.write(annotated)
        if args.show:
            cv2.imshow("YOLOv8 Object Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\nUnique objects tracked per class:")
    for class_name, ids in sorted(seen_ids.items()):
        print(f"  {class_name}: {len(ids)}")
    if args.output:
        print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
