# 🧠 Campaign Intelligence Platform

> **GenAI-powered customer response prediction and marketing analytics dashboard**
> Built for portfolio showcase — enterprise-grade, production-style, full-stack AI application.

---

## 🖥️ Tech Stack

| Layer      | Technology |
|------------|------------|
| Backend    | FastAPI, Python 3.11, Uvicorn |
| ML         | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| GenAI      | Hugging Face Inference API (Mistral-7B-Instruct) |
| Frontend   | React 18, Vite, Tailwind CSS |
| 3D         | React Three Fiber, Drei, Three.js |
| Animation  | Framer Motion |
| Charts     | Recharts |
| State      | Zustand |

---

## 📁 Folder Structure

```
campaign_intelligence/
├── backend/
│   ├── app.py                    # FastAPI entry point
│   ├── requirements.txt
│   ├── marketing_campaign.csv    # Dataset (tab-separated)
│   ├── .env.example
│   ├── models/                   # Auto-generated after first run
│   │   ├── best_model.pkl
│   │   ├── preprocessor.pkl
│   │   ├── metrics.json
│   │   └── feature_importance.json
│   ├── routes/
│   │   ├── summary.py            # GET /api/summary, /api/segment
│   │   ├── predict.py            # POST /api/predict-customer, /api/what-if
│   │   └── insights.py           # GET /api/model-metrics, /api/feature-importance
│   └── services/
│       ├── data_loader.py        # Tab-sep CSV loading (cached)
│       ├── feature_engineering.py # Age, tenure, spend aggregates, etc.
│       ├── preprocess.py         # sklearn ColumnTransformer pipeline
│       ├── model_training.py     # LR + RF + XGBoost, SMOTE, best model
│       ├── predict_service.py    # Single + batch prediction + feature drivers
│       ├── explain_service.py    # Segment stats, overview analytics
│       └── hf_service.py        # Hugging Face API prompting service
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── src/
│       ├── App.jsx               # Router + bootstrap data loading
│       ├── store.js              # Zustand global state
│       ├── index.css             # Tailwind + custom classes
│       ├── services/
│       │   └── api.js            # Axios service layer
│       ├── components/
│       │   ├── layout/
│       │   │   ├── Sidebar.jsx   # Dark animated sidebar
│       │   │   └── PageHeader.jsx
│       │   ├── ui/
│       │   │   ├── StatCard.jsx         # Animated KPI card
│       │   │   ├── AITextBox.jsx        # Typewriter AI output
│       │   │   ├── ProbabilityGauge.jsx # Circular SVG gauge
│       │   │   └── ParticleBackground.jsx # Canvas particle system
│       │   ├── charts/
│       │   │   └── Charts.jsx    # All Recharts components
│       │   └── three/
│       │       └── HeroScene.jsx # R3F 3D orb hero scene
│       └── pages/
│           ├── Overview.jsx           # KPIs + 3D hero + distribution charts
│           ├── CustomerIntelligence.jsx # Table + detail panel + AI prediction
│           ├── SegmentExplorer.jsx    # Filters + radar + AI segment summary
│           ├── WhatIfSimulator.jsx    # Sliders + before/after prediction
│           ├── ModelInsights.jsx      # Metrics + ROC + confusion matrix + FI
│           └── AIStrategyConsole.jsx  # Segment builder + AI campaign strategy
│
├── start.sh                      # Unified startup script
├── .replit
└── replit.nix
```

---

## 🚀 Running on Replit

### Step 1 — Import the project
Upload all files to a new Replit (Python/Node Repl or Blank Repl).

### Step 2 — Add your Hugging Face API key
1. Go to **Secrets** (lock icon in Replit sidebar)
2. Add secret: `HUGGINGFACE_API_KEY` = `hf_your_token_here`
3. Get a free token at: https://huggingface.co/settings/tokens

> **Without the key**, the app still works fully — AI text boxes will show pre-written analytical fallback responses.

### Step 3 — Run
Click the **Run** button. The `start.sh` script will:
1. Install Python dependencies
2. Start FastAPI on port 8000
3. Install Node dependencies
4. Start the React dev server on port 5173

### Step 4 — Wait for model training (~60 seconds)
The first run trains 3 ML models. The React UI loads immediately; data populates once training finishes. You'll see a loading toast.

---

## 🔧 Running Locally

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env          # add your HF key
python app.py                 # starts on :8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev                   # starts on :5173
```

Visit `http://localhost:5173`

API docs at `http://localhost:8000/docs`

---

## 🤖 ML Architecture

### Dataset
- 2,240 customers, 29 columns, tab-separated
- Target: `Response` (binary, ~15% positive — imbalanced)

### Feature Engineering
| Feature | Derivation |
|---------|-----------|
| `Age` | 2024 − Year_Birth |
| `Customer_Tenure_Days` | Reference date − Dt_Customer |
| `Total_Spent` | Sum of all Mnt columns |
| `Total_Purchases` | Web + Catalog + Store |
| `Children` | Kidhome + Teenhome |
| `AcceptedCampaignsTotal` | Sum AcceptedCmp1–5 |
| `CampaignEngagementRate` | AcceptedCampaignsTotal / 5 |
| `Avg_Spend_Per_Purchase` | Total_Spent / Total_Purchases |

### Models Trained
1. **Logistic Regression** — baseline, interpretable
2. **Random Forest** — 300 trees, balanced class weight
3. **XGBoost** — gradient boosting, scale_pos_weight for imbalance

### Class Imbalance Handling
SMOTE (Synthetic Minority Over-sampling Technique) applied to training data before fitting.

### Best Model Selection
Chosen by highest ROC-AUC on hold-out test set (80/20 stratified split).

---

## 🧬 GenAI Integration

The `hf_service.py` module calls the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/) with structured prompts for:

| Function | Output |
|----------|--------|
| `explain_prediction()` | Plain-English explanation of why a customer is classified as High/Low potential |
| `recommend_action()` | 3 specific retention/marketing actions for the customer |
| `generate_segment_summary()` | 4-sentence executive segment profile |
| `generate_campaign_strategy()` | Board-level campaign brief with channel and budget guidance |
| `explain_whatif()` | Analytical interpretation of a parameter change scenario |

**Default model**: `mistralai/Mistral-7B-Instruct-v0.2`
Override via `HF_MODEL` environment variable.

---

## 📊 API Endpoints

```
GET  /api/health               → system health + model status
GET  /api/summary              → full dataset overview stats
GET  /api/customers            → customer records (paginated)
GET  /api/segment              → filtered segment analytics
GET  /api/model-metrics        → all model comparison metrics
GET  /api/feature-importance   → top feature importances
POST /api/predict-customer     → single customer prediction
POST /api/predict-batch        → batch prediction list
POST /api/ai-explanation       → AI explanation + action (HF)
POST /api/what-if              → before/after scenario comparison
POST /api/segment-summary      → AI segment executive summary (HF)
POST /api/strategy             → AI campaign strategy brief (HF)
POST /api/train                → trigger model retraining (background)
```

---

## 🎨 UI Pages

| Page | Route | Purpose |
|------|-------|---------|
| Overview | `/overview` | 3D hero, KPI cards, spend/campaign charts |
| Customer Intel | `/customers` | Searchable table, detail panel, AI prediction |
| Segment Explorer | `/segments` | Filter-driven segment analysis + radar chart |
| What-If Lab | `/whatif` | Slider simulator with probability delta |
| Model Insights | `/model` | Metrics table, ROC curve, confusion matrix, feature importance |
| AI Strategy Console | `/strategy` | Segment builder + executive AI strategy generator |
<img width="757" height="642" alt="Screenshot 2026-04-20 at 13 48 52" src="https://github.com/user-attachments/assets/658440b8-988f-490f-af16-5a3ddde7d708" />
<img width="757" height="642" alt="Screenshot 2026-04-20 at 13 49 49" src="https://github.com/user-attachments/assets/3ffb8ec0-db02-4437-89db-9aa48539e14b" />



---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGINGFACE_API_KEY` | No | — | HF Inference API key (free tier available) |
| `HF_MODEL` | No | `mistralai/Mistral-7B-Instruct-v0.2` | HF model to use |
| `PORT` | No | `8000` | Backend port |

---

## ⚡ Performance Notes

- Dataset loads and caches on first request (LRU cache)
- Model artifacts persist to `backend/models/` and are reloaded on startup
- Retraining via `POST /api/train` runs in background (non-blocking)
- SMOTE + XGBoost training takes ~45–90 seconds on first run
- All HF API calls are async with 30s timeout; fall back to canned responses on failure

---

*Built as a portfolio-grade GenAI × ML × full-stack showcase.*
