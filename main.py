import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Industrial IoT Predictive Maintenance API",
    description="Real-time mechanical telemetry ingestion and Pandas-driven anomaly detection",
    version="1.0.0"
)

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        port=os.getenv("DB_PORT", "5432"),
        cursor_factory=RealDictCursor
    )

class TelemetryPayload(BaseModel):
    asset_id: int
    temperature_c: float
    vibration_mm_s: float
    current_amps: float

def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS iot_assets (
                asset_id SERIAL PRIMARY KEY,
                asset_name VARCHAR(120) NOT NULL,
                department VARCHAR(100),
                max_temp_c NUMERIC(5,2),
                max_vib_mm_s NUMERIC(5,2)
            );
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensor_telemetry (
                reading_id SERIAL PRIMARY KEY,
                asset_id INTEGER REFERENCES iot_assets(asset_id) ON DELETE CASCADE,
                temperature_c NUMERIC(6,2),
                vibration_mm_s NUMERIC(6,2),
                current_amps NUMERIC(6,2),
                health_score NUMERIC(5,2),
                maintenance_status VARCHAR(50),
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cur.execute("SELECT COUNT(*) as count FROM iot_assets;")
        row = cur.fetchone()
        if row and row["count"] == 0:
            cur.execute("""
                INSERT INTO iot_assets (asset_name, department, max_temp_c, max_vib_mm_s) VALUES
                ('Kiln A Circulation Blower Fan (3-Phase)', 'Fish Tech Lab 1', 85.0, 4.5),
                ('Kiln B Exhaust Draft Motor', 'Fish Tech Lab 2', 90.0, 4.5),
                ('Cold Room Bitzer Compressor Unit 1', 'Preservation Wing', 75.0, 3.5),
                ('Fish Meal Hammer Mill Drive Shaft', 'Feed Processing Plant', 80.0, 5.5);
            """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Python IoT Database Initialized Successfully")
    except Exception as e:
        print("⚠️ Database initialization error:", e)

init_db()

@app.get("/api/assets")
def get_assets():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM iot_assets ORDER BY asset_id ASC;")
    assets = cur.fetchall()
    cur.close()
    conn.close()
    return assets

@app.get("/api/telemetry")
def get_telemetry():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT t.*, a.asset_name, a.department
        FROM sensor_telemetry t
        INNER JOIN iot_assets a ON t.asset_id = a.asset_id
        ORDER BY t.reading_id DESC
        LIMIT 50;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {
            "kpis": {
                "total_readings": 0,
                "mean_health": 100.0,
                "critical_alerts": 0,
                "peak_vibration": 0.0
            },
            "readings": []
        }

    df = pd.DataFrame(rows)
    df["health_score"] = pd.to_numeric(df["health_score"])
    df["vibration_mm_s"] = pd.to_numeric(df["vibration_mm_s"])

    kpis = {
        "total_readings": int(len(df)),
        "mean_health": round(float(df["health_score"].mean()), 1),
        "critical_alerts": int((df["maintenance_status"] == "CRITICAL - INSPECT BEARING/COIL").sum()),
        "peak_vibration": round(float(df["vibration_mm_s"].max()), 2)
    }

    return {"kpis": kpis, "readings": rows}

@app.post("/api/telemetry")
def ingest_telemetry(payload: TelemetryPayload):
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT * FROM iot_assets WHERE asset_id = %s;", (payload.asset_id,))
    asset = cur.fetchone()
    if not asset:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Asset not found")

    max_temp = float(asset["max_temp_c"])
    max_vib = float(asset["max_vib_mm_s"])

    temp_ratio = max(0.0, (payload.temperature_c - 40.0) / (max_temp - 40.0))
    vib_ratio = max(0.0, payload.vibration_mm_s / max_vib)
    
    penalty = (vib_ratio * 60.0) + (temp_ratio * 40.0)
    health_score = round(float(np.clip(100.0 - penalty, 0.0, 100.0)), 1)

    if health_score < 50.0 or payload.vibration_mm_s >= max_vib * 1.25:
        status = "CRITICAL - INSPECT BEARING/COIL"
    elif health_score < 75.0 or payload.vibration_mm_s >= max_vib * 0.9:
        status = "WARNING - SCHEDULE LUBRICATION"
    else:
        status = "OPTIMAL"

    cur.execute("""
        INSERT INTO sensor_telemetry (
            asset_id, temperature_c, vibration_mm_s, current_amps, health_score, maintenance_status
        ) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *;
    """, (
        payload.asset_id,
        payload.temperature_c,
        payload.vibration_mm_s,
        payload.current_amps,
        health_score,
        status
    ))
    new_reading = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return new_reading

@app.delete("/api/telemetry/{reading_id}")
def delete_reading(reading_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM sensor_telemetry WHERE reading_id = %s;", (reading_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Deleted"}

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Python FastAPI — Industrial IoT & Predictive Maintenance</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #090d16; --card: #111827; --border: #1e293b;
      --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8;
      --green: #34d399; --amber: #fbbf24; --red: #f87171;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 28px 42px; }
    .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .pill {
      background: rgba(56, 189, 248, 0.15); color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.35); font-size: 11px;
      font-weight: 700; padding: 5px 12px; border-radius: 999px; letter-spacing: 0.7px;
    }
    .docs-link { color: var(--green); font-size: 13px; text-decoration: none; font-weight: 600; }
    h1 { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
    .sub { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; margin-bottom: 24px; }
    .kpi { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
    .kpi-label { font-size: 11px; color: var(--muted); text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }
    .kpi-val { font-size: 26px; font-weight: 800; color: var(--accent); }
    .main-grid { display: grid; grid-template-columns: 360px 1fr; gap: 22px; align-items: start; }
    .right-col { display: flex; flex-direction: column; gap: 22px; }
    .panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 22px; }
    .panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
    .panel h2 { font-size: 16px; font-weight: 700; }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
    input, select {
      width: 100%; background: #060911; border: 1px solid var(--border);
      color: var(--text); padding: 10px; border-radius: 8px; font-size: 14px; margin-bottom: 14px;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .btn {
      width: 100%; background: #0284c7; color: #fff; border: none; padding: 11px;
      border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 13px; margin-bottom: 10px;
    }
    .btn-sim { background: rgba(251, 191, 36, 0.15); color: var(--amber); border: 1px solid rgba(251, 191, 36, 0.4); }
    .btn-export {
      background: rgba(52, 211, 153, 0.15); color: var(--green);
      border: 1px solid rgba(52, 211, 153, 0.4); padding: 7px 12px;
      border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    th { background: #0b1120; color: var(--muted); text-align: left; padding: 10px; font-size: 11px; text-transform: uppercase; }
    td { padding: 12px 10px; border-bottom: 1px solid var(--border); }
    .badge { padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }
    .OPTIMAL { background: rgba(52, 211, 153, 0.15); color: var(--green); }
    .WARNING { background: rgba(251, 191, 36, 0.15); color: var(--amber); }
    .CRITICAL { background: rgba(248, 113, 113, 0.15); color: var(--red); }
    .chart-box { position: relative; height: 250px; width: 100%; }
  </style>
</head>
<body>
  <div class="top-bar">
    <span class="pill">PYTHON 3 + FASTAPI + PANDAS + NEON POSTGRESQL</span>
    <a href="/docs" target="_blank" class="docs-link">⚡ View Interactive FastAPI Swagger Docs (/docs) ➔</a>
  </div>
  <h1>Industrial IoT Sensor & Predictive Maintenance Telemetry ⚙️🐍</h1>
  <p class="sub">Real-time mechanical vibration (ISO 10816), motor current, and thermal anomaly detection powered by Python Pandas</p>

  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-label">TOTAL SENSOR PACKETS</div><div class="kpi-val" id="kpi-total">0</div></div>
    <div class="kpi"><div class="kpi-label">MEAN FLEET HEALTH SCORE</div><div class="kpi-val" id="kpi-health">100.0 %</div></div>
    <div class="kpi"><div class="kpi-label">PEAK VIBRATION RMS</div><div class="kpi-val" id="kpi-vib">0.00 mm/s</div></div>
    <div class="kpi"><div class="kpi-label">CRITICAL BEARING/COIL ALERTS</div><div class="kpi-val" id="kpi-crit" style="color:var(--red);">0</div></div>
  </div>

  <div class="main-grid">
    <div class="panel">
      <div class="panel-head"><h2>Ingest IoT Sensor Packet</h2></div>
      <form id="iot-form">
        <label>Target Electro-Mechanical Asset</label>
        <select id="asset_id" required></select>
        <div class="row">
          <div><label>Bearing Temp (°C)</label><input type="number" step="0.1" id="temperature_c" value="62.5" required /></div>
          <div><label>Vibration RMS (mm/s)</label><input type="number" step="0.1" id="vibration_mm_s" value="2.1" required /></div>
        </div>
        <label>Motor Phase Current Draw (Amps)</label>
        <input type="number" step="0.1" id="current_amps" value="14.2" required />
        <button type="submit" class="btn">Transmit Telemetry to Python Engine ➔</button>
        <button type="button" class="btn btn-sim" onclick="simulateAnomaly()">⚠️ Simulate Bearing Wear Spike</button>
      </form>
    </div>

    <div class="right-col">
      <div class="panel">
        <div class="panel-head">
          <h2>Vibration RMS (mm/s) vs. Machine Health Index (%)</h2>
        </div>
        <div class="chart-box"><canvas id="iotChart"></canvas></div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <h2>Live Predictive Maintenance Log</h2>
          <button class="btn-export" onclick="exportCSV()">⬇ Export CSV</button>
        </div>
        <table>
          <thead>
            <tr>
              <th>ID</th><th>ASSET</th><th>TEMP (°C)</th><th>VIB (MM/S)</th>
              <th>CURRENT (A)</th><th>HEALTH</th><th>PREDICTIVE DIAGNOSTIC</th>
            </tr>
          </thead>
          <tbody id="log-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let cachedReadings = [];
    let chartInstance = null;

    async function loadAssets() {
      const res = await fetch('/api/assets');
      const assets = await res.json();
      const sel = document.getElementById('asset_id');
      sel.innerHTML = '';
      assets.forEach(a => {
        const opt = document.createElement('option');
        opt.value = a.asset_id;
        opt.textContent = a.asset_name + ' [' + a.department + ']';
        sel.appendChild(opt);
      });
    }

    async function loadTelemetry() {
      const res = await fetch('/api/telemetry');
      const data = await res.json();
      cachedReadings = data.readings;

      document.getElementById('kpi-total').textContent = data.kpis.total_readings;
      document.getElementById('kpi-health').textContent = data.kpis.mean_health + ' %';
      document.getElementById('kpi-vib').textContent = data.kpis.peak_vibration + ' mm/s';
      document.getElementById('kpi-crit').textContent = data.kpis.critical_alerts;

      const tbody = document.getElementById('log-body');
      tbody.innerHTML = '';
      cachedReadings.forEach(r => {
        const badgeClass = r.maintenance_status.split(' ')[0];
        const tr = document.createElement('tr');
        tr.innerHTML =
          '<td>#' + r.reading_id + '</td>' +
          '<td><strong>' + r.asset_name + '</strong></td>' +
          '<td>' + r.temperature_c + ' °C</td>' +
          '<td>' + r.vibration_mm_s + ' mm/s</td>' +
          '<td>' + r.current_amps + ' A</td>' +
          '<td style="font-weight:700;color:#38bdf8;">' + r.health_score + '%</td>' +
          '<td><span class="badge ' + badgeClass + '">' + r.maintenance_status + '</span></td>';
        tbody.appendChild(tr);
      });

      renderChart(cachedReadings);
    }

    function renderChart(readings) {
      const ctx = document.getElementById('iotChart').getContext('2d');
      const chron = [...readings].reverse();
      const labels = chron.map(r => '#' + r.reading_id + ' ' + r.asset_name.slice(0, 15));
      const vibs = chron.map(r => parseFloat(r.vibration_mm_s));
      const health = chron.map(r => parseFloat(r.health_score));

      if (chartInstance) chartInstance.destroy();
      chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            { label: 'Vibration RMS (mm/s)', data: vibs, backgroundColor: 'rgba(248, 113, 113, 0.65)', yAxisID: 'y' },
            { label: 'Health Score (%)', data: health, type: 'line', borderColor: '#34d399', tension: 0.3, yAxisID: 'y1' }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#cbd5e1' } } },
          scales: {
            x: { ticks: { color: '#94a3b8' } },
            y: { position: 'left', title: { display: true, text: 'Vibration (mm/s)', color: '#f87171' }, ticks: { color: '#94a3b8' } },
            y1: { position: 'right', min: 0, max: 100, title: { display: true, text: 'Health (%)', color: '#34d399' }, ticks: { color: '#94a3b8' }, grid: { drawOnChartArea: false } }
          }
        }
      });
    }

    document.getElementById('iot-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      await sendPacket({
        asset_id: parseInt(document.getElementById('asset_id').value),
        temperature_c: parseFloat(document.getElementById('temperature_c').value),
        vibration_mm_s: parseFloat(document.getElementById('vibration_mm_s').value),
        current_amps: parseFloat(document.getElementById('current_amps').value)
      });
    });

    async function simulateAnomaly() {
      document.getElementById('temperature_c').value = (88 + Math.random() * 12).toFixed(1);
      document.getElementById('vibration_mm_s').value = (5.8 + Math.random() * 2.5).toFixed(2);
      document.getElementById('current_amps').value = (21 + Math.random() * 5).toFixed(1);
      document.getElementById('iot-form').requestSubmit();
    }

    async function sendPacket(payload) {
      await fetch('/api/telemetry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      await loadTelemetry();
    }

    function exportCSV() {
      if (!cachedReadings.length) return alert('No telemetry records yet.');
      const headers = ['Reading ID','Asset Name','Department','Temperature (C)','Vibration (mm/s)','Current (A)','Health Score (%)','Diagnostic Status','Timestamp'];
      const rows = cachedReadings.map(r => [
        r.reading_id, '"' + r.asset_name + '"', '"' + r.department + '"',
        r.temperature_c, r.vibration_mm_s, r.current_amps, r.health_score,
        '"' + r.maintenance_status + '"', '"' + r.recorded_at + '"'
      ]);
      const csv = [headers.join(','), ...rows.map(row => row.join(','))].join('\\n');
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
      a.download = 'iot_predictive_maintenance_log.csv';
      a.click();
    }

    loadAssets();
    loadTelemetry();
  </script>
</body>
</html>"""