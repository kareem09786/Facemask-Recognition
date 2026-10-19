# Face Mask Recognition

A pipeline that trains a CNN to tell whether a face image shows a mask, then uses it to label faces in a photo or a live webcam feed.

## How it works
1. **`train.py`** loads labelled images, makes a stratified 70/15/15 train/validation/test split, trains a small CNN with data augmentation (flip, rotation, zoom) and early stopping, and evaluates on the held-out test set.
2. **`detect.py`** finds faces with an OpenCV Haar cascade, crops each one, classifies it with the trained model, and draws a green (mask) or red (no mask) box with the confidence.

## Dataset
Not included. Put any labelled face-mask image set in this layout:
```
dataset/
  with_mask/      *.jpg
  without_mask/   *.jpg
```

## Run it
```bash
pip install -r requirements.txt
python train.py                      # trains, saves mask_cnn.keras and results/
python detect.py                     # live webcam, press q to quit
python detect.py --image photo.jpg   # single image, saves photo_out.jpg
```

## Outputs
`train.py` prints a per-class precision/recall report for the test set and writes `results/metrics.json`, `results/confusion_matrix.png` and `results/training_curve.png`. Check the recall for `without_mask` in particular: the costly mistake in this task is calling an unmasked face masked.

## Model
Three `Conv2D` + `MaxPooling2D` blocks (16, 32, 64 filters), dropout, a 64-unit dense layer and a sigmoid output giving the probability of a mask. Images are resized to 128×128 and rescaled inside the model.

## Limitations
- Haar cascades work best on frontal, well-lit faces
- Accuracy depends entirely on the training data. A model trained on a small or uniform set will not generalise to other lighting, angles or mask types
- Not tested as a safety or compliance tool

## Related
[Mask-recognition-using-Convolutional-Neural-Network](https://github.com/kareem09786/Mask-recognition-using-Convolutional-Neural-Network) contains the original exploratory notebook.
