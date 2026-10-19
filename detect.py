"""Detect faces and classify each as masked / unmasked using the trained model.

Usage:
    python detect.py                    # live webcam, press q to quit
    python detect.py --image photo.jpg  # one image; saves photo_out.jpg
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

GREEN, RED = (0, 180, 0), (0, 0, 220)


def annotate(frame, model, cascade, img_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        crop = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (img_size, img_size)).astype("float32")
        p_mask = float(model.predict(crop[None], verbose=0)[0][0])  # model outputs P(with_mask)
        has_mask = p_mask >= 0.5
        label = f"Mask {p_mask:.0%}" if has_mask else f"No mask {1 - p_mask:.0%}"
        color = GREEN if has_mask else RED
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, len(faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mask_cnn.keras")
    ap.add_argument("--image")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    img_size = model.input_shape[1]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        frame, n = annotate(frame, model, cascade, img_size)
        out = os.path.splitext(args.image)[0] + "_out.jpg"
        cv2.imwrite(out, frame)
        print(f"{n} face(s) found. Saved {out}")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, _ = annotate(frame, model, cascade, img_size)
        cv2.imshow("Face mask detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
