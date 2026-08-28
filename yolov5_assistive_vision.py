"""Real-time YOLOv5 object detection with relative object positions.

The camera image is divided into a 3x3 grid. Each detected object's bounding-box
centre is classified as top-left, centre, bottom-right, and so on.

Controls:
    q or Esc  Quit
    s         Toggle spoken announcements
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import defaultdict

import cv2
import torch


POSITION_COLOURS = {
    "top-left": (255, 150, 0),
    "top-centre": (255, 200, 0),
    "top-right": (255, 150, 0),
    "middle-left": (0, 200, 255),
    "centre": (0, 255, 0),
    "middle-right": (0, 200, 255),
    "bottom-left": (255, 0, 180),
    "bottom-centre": (180, 0, 255),
    "bottom-right": (255, 0, 180),
}


def get_position(box: tuple[int, int, int, int], frame_width: int, frame_height: int) -> str:
    """Return the 3x3-grid position of the bounding-box centre."""
    x1, y1, x2, y2 = box
    centre_x = (x1 + x2) / 2
    centre_y = (y1 + y2) / 2

    if centre_x < frame_width / 3:
        horizontal = "left"
    elif centre_x < 2 * frame_width / 3:
        horizontal = "centre"
    else:
        horizontal = "right"

    if centre_y < frame_height / 3:
        vertical = "top"
    elif centre_y < 2 * frame_height / 3:
        vertical = "middle"
    else:
        vertical = "bottom"

    if vertical == "middle" and horizontal == "centre":
        return "centre"
    return f"{vertical}-{horizontal}"


class SpeechWorker:
    """Non-blocking text-to-speech worker; gracefully disables itself if unavailable."""

    def __init__(self) -> None:
        self.messages: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self.available = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            while True:
                message = self.messages.get()
                if message is None:
                    break
                engine.say(message)
                engine.runAndWait()
        except Exception as error:
            self.available = False
            print(f"Speech disabled: {error}")

    def say(self, message: str) -> None:
        if self.available and not self.messages.full():
            self.messages.put_nowait(message)

    def close(self) -> None:
        if self.available:
            try:
                self.messages.put_nowait(None)
            except queue.Full:
                pass


def draw_grid(frame) -> None:
    """Draw the nine position regions on the camera image."""
    height, width = frame.shape[:2]
    colour = (100, 100, 100)
    for x in (width // 3, 2 * width // 3):
        cv2.line(frame, (x, 0), (x, height), colour, 1)
    for y in (height // 3, 2 * height // 3):
        cv2.line(frame, (0, y), (width, y), colour, 1)


def make_announcement(detections: list[tuple[str, str, float]]) -> str:
    """Create a short spoken summary, avoiding repeated identical objects."""
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for label, position, _confidence in detections:
        grouped[(label, position)] += 1

    phrases = []
    for (label, position), count in grouped.items():
        if count == 1:
            phrases.append(f"{label} at {position.replace('-', ' ')}")
        else:
            phrases.append(f"{count} {label}s at {position.replace('-', ' ')}")
    return ", ".join(phrases[:5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv5 assistive-vision camera")
    parser.add_argument("--camera", type=int, default=0, help="camera device number")
    parser.add_argument("--model", default="yolov5s", choices=["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"])
    parser.add_argument("--confidence", type=float, default=0.45, help="minimum confidence, 0 to 1")
    parser.add_argument("--speech-interval", type=float, default=3.0, help="seconds between announcements")
    parser.add_argument("--no-speech", action="store_true", help="start with speech disabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    model = torch.hub.load(
        "./yolov5",
        "custom",
        path="./yolov5/runs/train/vision3_gpu/weights/best.pt",
        source="local",
    )
    model.to(device)
    model.conf = args.confidence

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    speaker = SpeechWorker()
    speech_enabled = not args.no_speech
    last_spoken_at = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a camera frame.")
                break

            frame_height, frame_width = frame.shape[:2]
            results = model(frame)
            detections: list[tuple[str, str, float]] = []

            # Each row: x1, y1, x2, y2, confidence, class_id
            for x1, y1, x2, y2, confidence, class_id in results.xyxy[0].cpu().numpy():
                box = (int(x1), int(y1), int(x2), int(y2))
                label = model.names[int(class_id)]
                position = get_position(box, frame_width, frame_height)
                detections.append((label, position, float(confidence)))

                colour = POSITION_COLOURS[position]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colour, 2)
                text = f"{label} | {position} | {confidence:.0%}"
                text_y = max(24, box[1] - 8)
                cv2.putText(frame, text, (box[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2)

                object_x = (box[0] + box[2]) // 2
                object_y = (box[1] + box[3]) // 2
                cv2.circle(frame, (object_x, object_y), 5, colour, -1)

            draw_grid(frame)
            status = f"Detections: {len(detections)} | Speech: {'ON' if speech_enabled else 'OFF'}"
            cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            now = time.monotonic()
            if speech_enabled and detections and now - last_spoken_at >= args.speech_interval:
                announcement = make_announcement(detections)
                if announcement:
                    speaker.say(announcement)
                    last_spoken_at = now

            cv2.imshow("YOLOv5 Assistive Vision - press Q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                speech_enabled = not speech_enabled

    finally:
        camera.release()
        cv2.destroyAllWindows()
        speaker.close()


if __name__ == "__main__":
    main()
