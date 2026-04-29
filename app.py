"""
Bot Detector — Flask Backend (v2 — Circle Task)
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import os

try:
    from tensorflow import keras
    MODEL_AVAILABLE = os.path.exists('bot_detector_model.keras')
    if MODEL_AVAILABLE:
        model = keras.models.load_model('bot_detector_model.keras')
        print('[OK] Model loaded: bot_detector_model.keras')
    else:
        model = None
        print('[WARN] Model file not found — running in DEMO mode')
except ImportError:
    model = None
    MODEL_AVAILABLE = False
    print('[WARN] TensorFlow not installed — running in DEMO mode')

MAX_LEN   = 200
FEATURES  = 4
EVENT_MAP = {'move': 0, 'click': 1, 'release': 2}

app = Flask(__name__)

def preprocess_events(events):
    if not events:
        return np.zeros((1, MAX_LEN, FEATURES), dtype=np.float32)

    arr = []
    t_vals = [e['t'] for e in events]
    t_min, t_max = min(t_vals), max(t_vals)
    t_range = max(t_max - t_min, 1)

    xs = [e['x'] for e in events]
    ys = [e['y'] for e in events]
    w  = max(xs) if max(xs) > 0 else 1920
    h  = max(ys) if max(ys) > 0 else 1080

    for e in events:
        t_norm = (e['t'] - t_min) / t_range
        ev_enc = EVENT_MAP.get(e.get('type', 'move'), 0)
        x_norm = e['x'] / w
        y_norm = e['y'] / h
        arr.append([t_norm, ev_enc, x_norm, y_norm])

    arr = np.array(arr, dtype=np.float32)
    seq_len = len(arr)
    if seq_len < MAX_LEN:
        pad = np.zeros((MAX_LEN - seq_len, FEATURES), dtype=np.float32)
        arr = np.vstack([arr, pad])
    else:
        arr = arr[:MAX_LEN]

    return arr[np.newaxis, ...]

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json(force=True)
    events = data.get('events', [])

    if len(events) < 10:
        return jsonify({'error': 'Not enough data — need at least 10 mouse events'}), 400

    X = preprocess_events(events)

    if model is not None:
        bot_prob = float(model.predict(X, verbose=0)[0][0])
    else:
        np.random.seed(len(events) % 99)
        bot_prob = float(np.clip(np.random.beta(2, 5), 0.01, 0.99))

    human_prob = 1.0 - bot_prob
    verdict    = 'BOT' if bot_prob >= 0.5 else 'HUMAN'
    confidence = max(bot_prob, human_prob) * 100

    return jsonify({
        'verdict':     verdict,
        'bot_pct':     round(bot_prob * 100, 1),
        'human_pct':   round(human_prob * 100, 1),
        'confidence':  round(confidence, 1),
        'events_used': min(len(events), MAX_LEN),
        'demo_mode':   model is None,
    })


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bot Detector — Circle Task</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }

    h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
    h1 span { color: #38bdf8; }
    .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 24px; text-align: center; }

    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 16px;
      padding: 24px;
      width: 100%;
      max-width: 700px;
    }

    #canvas {
      width: 100%;
      height: 380px;
      background: #0f172a;
      border: 2px solid #334155;
      border-radius: 12px;
      cursor: crosshair;
      display: block;
    }

    .canvas-label {
      text-align: center;
      color: #64748b;
      font-size: 0.85rem;
      margin-top: 8px;
      min-height: 20px;
    }

    .progress-wrap { margin: 16px 0 12px; }
    .progress-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: #94a3b8;
      margin-bottom: 6px;
    }
    .progress-bar-bg { height: 8px; background: #334155; border-radius: 99px; overflow: hidden; }
    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #0ea5e9, #38bdf8);
      border-radius: 99px;
      width: 0%;
      transition: width 0.3s ease;
    }

    .btn-row { display: flex; gap: 10px; margin-top: 14px; }
    button {
      flex: 1; padding: 11px 20px; border: none; border-radius: 10px;
      font-size: 0.95rem; font-weight: 600; cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
    }
    button:active { transform: scale(0.97); }
    #analyzeBtn { background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white; }
    #analyzeBtn:disabled { opacity: 0.4; cursor: not-allowed; }
    #resetBtn { background: #334155; color: #94a3b8; flex: 0 0 auto; padding: 11px 16px; }

    #result { margin-top: 20px; display: none; animation: fadeIn 0.4s ease; }
    @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }

    .verdict-row { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
    .verdict-badge {
      font-size: 1.4rem; font-weight: 800;
      padding: 6px 18px; border-radius: 99px; letter-spacing: 1px;
    }
    .verdict-badge.human { background: #064e3b; color: #34d399; }
    .verdict-badge.bot   { background: #450a0a; color: #f87171; }
    .verdict-sub { color: #94a3b8; font-size: 0.85rem; }

    .bars { display: flex; flex-direction: column; gap: 10px; }
    .bar-item label { display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px; }
    .bar-item label span:first-child { color: #cbd5e1; }
    .bar-item label span:last-child  { font-weight: 700; }
    .bar-track { height: 14px; background: #0f172a; border-radius: 99px; overflow: hidden; }
    .bar-fill-human { height:100%; background: linear-gradient(90deg,#059669,#34d399); border-radius:99px; width:0%; transition: width 0.8s cubic-bezier(.17,.67,.38,1.2); }
    .bar-fill-bot   { height:100%; background: linear-gradient(90deg,#dc2626,#f87171); border-radius:99px; width:0%; transition: width 0.8s cubic-bezier(.17,.67,.38,1.2); }
    .meta { margin-top: 12px; font-size: 0.78rem; color: #475569; text-align: right; }
  </style>
</head>
<body>

<h1>Mouse <span>Bot Detector</span></h1>
<p class="subtitle">Follow the glowing dot around the circle — the AI will analyse your movement.</p>

<div class="card">
  <canvas id="canvas"></canvas>
  <p class="canvas-label" id="canvasLabel">👆 Follow the blue dot around the circle to begin</p>

  <div class="progress-wrap">
    <div class="progress-header">
      <span>Events captured</span>
      <span id="countLabel">0 / 200</span>
    </div>
    <div class="progress-bar-bg">
      <div class="progress-bar-fill" id="progressFill"></div>
    </div>
  </div>

  <div class="btn-row">
    <button id="analyzeBtn" disabled>🔍 Analyse Now</button>
    <button id="resetBtn">↺ Reset</button>
  </div>

  <div id="result">
    <div class="verdict-row">
      <div class="verdict-badge" id="verdictBadge">—</div>
      <div class="verdict-sub"  id="verdictSub"></div>
    </div>
    <div class="bars">
      <div class="bar-item">
        <label><span>🧑 Human probability</span><span id="humanPct">—</span></label>
        <div class="bar-track"><div class="bar-fill-human" id="humanBar"></div></div>
      </div>
      <div class="bar-item">
        <label><span>🤖 Bot probability</span><span id="botPct">—</span></label>
        <div class="bar-track"><div class="bar-fill-bot" id="botBar"></div></div>
      </div>
    </div>
    <p class="meta" id="metaText"></p>
  </div>
</div>

<script>
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const progressFill= document.getElementById('progressFill');
const countLabel  = document.getElementById('countLabel');
const analyzeBtn  = document.getElementById('analyzeBtn');
const resetBtn    = document.getElementById('resetBtn');
const canvasLabel = document.getElementById('canvasLabel');

const MAX_EVENTS  = 200;
const MIN_ANALYZE = 50;
const GUIDE_SPEED = 0.02;   // radians per frame

let events     = [];
let analyzing  = false;
let guideAngle = 0;
let animId     = null;

// ── Sizing ─────────────────────────────────────────────────────────
function resizeCanvas() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

function getCircle() {
  const cx = canvas.width  / 2;
  const cy = canvas.height / 2;
  const r  = Math.min(cx, cy) * 0.82;
  return { cx, cy, r };
}

// ── Animation loop ─────────────────────────────────────────────────
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const { cx, cy, r } = getCircle();

  // Faint guide circle
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = '#1e3a5f';
  ctx.lineWidth   = 3;
  ctx.setLineDash([8, 6]);
  ctx.stroke();
  ctx.setLineDash([]);

  // User trail
  if (events.length > 1) {
    for (let i = 1; i < events.length; i++) {
      const alpha = 0.15 + 0.85 * (i / events.length);
      ctx.beginPath();
      ctx.moveTo(events[i-1].cx, events[i-1].cy);
      ctx.lineTo(events[i].cx,   events[i].cy);
      ctx.strokeStyle = `rgba(56,189,248,${alpha})`;
      ctx.lineWidth   = 2;
      ctx.setLineDash([]);
      ctx.stroke();
    }
    // Cursor dot
    const last = events[events.length - 1];
    ctx.beginPath();
    ctx.arc(last.cx, last.cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.fill();
  }

  // Moving guide dot
  if (!analyzing) {
    const gx = cx + r * Math.cos(guideAngle);
    const gy = cy + r * Math.sin(guideAngle);

    // Glow
    const grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, 20);
    grad.addColorStop(0, 'rgba(56,189,248,0.5)');
    grad.addColorStop(1, 'rgba(56,189,248,0)');
    ctx.beginPath();
    ctx.arc(gx, gy, 20, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Core dot
    ctx.beginPath();
    ctx.arc(gx, gy, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.fill();

    // White centre
    ctx.beginPath();
    ctx.arc(gx, gy, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();

    guideAngle += GUIDE_SPEED;
  }

  animId = requestAnimationFrame(draw);
}
draw();

// ── Record events ──────────────────────────────────────────────────
canvas.addEventListener('mousemove',  e => record(e, 'move'));
canvas.addEventListener('mousedown',  e => record(e, 'click'));
canvas.addEventListener('mouseup',    e => record(e, 'release'));

function record(e, type) {
  if (analyzing) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;

  // Canvas pixel coords for drawing
  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top)  * scaleY;

  // Raw coords for backend (normalised there)
  const px = Math.round(e.clientX - rect.left);
  const py = Math.round(e.clientY - rect.top);

  events.push({ t: performance.now(), type, x: px, y: py, cx, cy });

  const pct = Math.min(events.length / MAX_EVENTS * 100, 100);
  progressFill.style.width = pct + '%';
  countLabel.textContent   = `${Math.min(events.length, MAX_EVENTS)} / ${MAX_EVENTS}`;

  if (events.length >= MIN_ANALYZE) {
    analyzeBtn.disabled = false;
    canvasLabel.textContent = events.length >= MAX_EVENTS
      ? '✅ Ready — click Analyse Now!'
      : `Keep tracing… (${events.length} events)`;
  } else {
    canvasLabel.textContent = `Follow the dot… (${events.length} events)`;
  }

  if (events.length >= MAX_EVENTS && !analyzing) analyse();
}

// ── Analyse ────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', analyse);

async function analyse() {
  if (analyzing || events.length < MIN_ANALYZE) return;
  analyzing = true;
  analyzeBtn.disabled    = true;
  analyzeBtn.textContent = '⏳ Analysing…';
  canvasLabel.textContent = 'Running AI model…';

  const payload = events.map(e => ({ t: e.t, type: e.type, x: e.x, y: e.y }));

  try {
    const resp = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ events: payload }),
    });
    const data = await resp.json();
    data.error ? (canvasLabel.textContent = '⚠️ ' + data.error) : showResult(data);
  } catch {
    canvasLabel.textContent = '❌ Could not reach server. Is app.py running?';
  }

  analyzeBtn.textContent = '🔍 Analyse Now';
  analyzeBtn.disabled    = false;
  analyzing              = false;
}

// ── Show result ────────────────────────────────────────────────────
function showResult(data) {
  const badge = document.getElementById('verdictBadge');
  badge.textContent = data.verdict;
  badge.className   = 'verdict-badge ' + data.verdict.toLowerCase();

  document.getElementById('verdictSub').innerHTML  = `Confidence: <strong>${data.confidence}%</strong>`;
  document.getElementById('humanPct').textContent  = data.human_pct + '%';
  document.getElementById('botPct').textContent    = data.bot_pct   + '%';
  document.getElementById('result').style.display  = 'block';
  document.getElementById('metaText').textContent  = `Based on ${data.events_used} mouse events`;
  canvasLabel.textContent = `Result: ${data.verdict} (${data.confidence}% confidence)`;

  setTimeout(() => {
    document.getElementById('humanBar').style.width = data.human_pct + '%';
    document.getElementById('botBar').style.width   = data.bot_pct   + '%';
  }, 50);
}

// ── Reset ──────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  events     = [];
  analyzing  = false;
  guideAngle = 0;
  analyzeBtn.disabled      = true;
  analyzeBtn.textContent   = '🔍 Analyse Now';
  progressFill.style.width = '0%';
  countLabel.textContent   = '0 / 200';
  canvasLabel.textContent  = '👆 Follow the blue dot around the circle to begin';
  document.getElementById('result').style.display    = 'none';
  document.getElementById('humanBar').style.width    = '0%';
  document.getElementById('botBar').style.width      = '0%';
});
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

if __name__ == '__main__':
    print('\n' + '='*50)
    print('  Bot Detector — Circle Task Demo')
    print('  Open: http://localhost:5000')
    print(f'  Model loaded: {MODEL_AVAILABLE}')
    print('='*50 + '\n')
    app.run(debug=False, port=5000)
