# 🚦 Road Sign Detection

Real-time road sign detection using YOLOv8s, trained on a traffic sign dataset from Kaggle.

**Live demo → [cv-road-sign-detecion.streamlit.app](https://cv-road-sign-detecion.streamlit.app)**

## Classes

The model detects 15 classes:

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Green Light | 8 | Speed Limit 40 |
| 1 | Red Light | 9 | Speed Limit 50 |
| 2 | Speed Limit 10 | 10 | Speed Limit 60 |
| 3 | Speed Limit 100 | 11 | Speed Limit 70 |
| 4 | Speed Limit 110 | 12 | Speed Limit 80 |
| 5 | Speed Limit 120 | 13 | Speed Limit 90 |
| 6 | Speed Limit 20 | 14 | Stop |
| 7 | Speed Limit 30 | | |

## Dataset

Trained on the [Car Detection dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection) from Kaggle.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `best.pt` in the same folder before running.

## Training

Trained with YOLOv8s for 50 epochs on Kaggle GPU (T4/P100).

- Input size: 640×640
- Confidence threshold: 0.25
