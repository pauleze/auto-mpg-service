# Auto MPG Prediction Microservice — Deployment Guide

## Files in this folder

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application (the service) |
| `generate_model.py` | Colab cell to re-train and export model weights |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Render.com auto-deploy config |

---

## Step 1 — Generate the model weights file

1. Open your Module 4 Google Colab notebook.
2. Paste the contents of `generate_model.py` into a new cell at the bottom.
3. Run it. It will print scaler statistics and save `auto_mpg_model.pth`.
4. **Download `auto_mpg_model.pth`** from Colab's file panel (left sidebar).
5. If the printed scaler statistics differ from those hardcoded in `main.py`,
   update `FEATURE_MEANS` and `FEATURE_STDS` in `main.py` to match.

---

## Step 2 — Push to GitHub

1. Create a new **public** GitHub repository (e.g. `auto-mpg-service`).
2. Add all files in this folder **plus `auto_mpg_model.pth`**:
   ```
   auto_mpg_service/
   ├── main.py
   ├── requirements.txt
   ├── render.yaml
   └── auto_mpg_model.pth   ← generated in Step 1
   ```
3. Commit and push.

---

## Step 3 — Deploy on Render (free)

1. Go to https://render.com and sign up (free tier works fine).
2. Click **New → Web Service**.
3. Connect your GitHub repository.
4. Render will detect `render.yaml` automatically. Confirm settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**.
6. Wait ~3 minutes for the build. Your URL will be:
   `https://auto-mpg-prediction-api.onrender.com`

---

## Step 4 — Test the service

### Health check (browser or curl)
```
GET https://auto-mpg-prediction-api.onrender.com/
```

### Predict (curl)
```bash
curl -X POST "https://auto-mpg-prediction-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cylinders": 4,
    "displacement": 140.0,
    "horsepower": 90.0,
    "weight": 2264,
    "acceleration": 15.5,
    "model_year": 71,
    "origin": 1
  }'
```

### Interactive docs
Visit: `https://auto-mpg-prediction-api.onrender.com/docs`

---

## Note on Render free tier

Render free services spin down after 15 minutes of inactivity and take ~30 seconds
to wake up on the first request. To avoid this for your instructor:
- Use a free uptime monitor (e.g. https://uptimerobot.com) to ping `/` every 10 minutes.
- Or upgrade to Render Starter ($7/month) for always-on.
