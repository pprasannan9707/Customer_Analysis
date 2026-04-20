#!/usr/bin/env bash
# Campaign Intelligence Platform — Replit startup script
set -e

echo "========================================"
echo "  Campaign Intelligence Platform"
echo "========================================"

# ── Backend setup ──────────────────────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
cd /home/runner/${REPL_SLUG}/backend
pip install -r requirements.txt -q --disable-pip-version-check

echo "[2/4] Starting FastAPI backend on :8000..."
python app.py &
BACKEND_PID=$!

# ── Frontend setup ─────────────────────────────────────────────────────────
echo "[3/4] Installing Node dependencies..."
cd /home/runner/${REPL_SLUG}/frontend
npm install --silent

echo "[4/4] Starting React frontend on :5173..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅  Backend  → http://localhost:8000"
echo "✅  Frontend → http://localhost:5173"
echo ""
echo "📊  API docs → http://localhost:8000/docs"
echo ""
echo "⏳  First run will train the ML model (~60 seconds)"
echo "    The UI is live immediately; data loads once training completes."
echo ""

# Keep alive
wait $BACKEND_PID $FRONTEND_PID
