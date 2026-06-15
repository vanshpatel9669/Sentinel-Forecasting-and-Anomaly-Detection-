# 🚀 Sentinel — Anomaly-Aware Forecasting & Monitoring Platform

An enterprise-grade decision intelligence platform that combines time-series forecasting, anomaly detection, operational risk scoring, and automated alert prioritization to transform raw business metrics into actionable insights.

Built using FastAPI, Python, Docker, and modular monitoring pipelines, Sentinel helps organizations proactively identify abnormal behavior, assess operational risk, and recommend corrective actions in real time.

---

## 🎯 Key Highlights

✅ Time-Series Forecasting Engine

✅ CUSUM & Z-Score Anomaly Detection

✅ Operational Risk Scoring

✅ Incident Priority Classification

✅ REST API Architecture

✅ Dockerized Deployment

✅ OpenAPI / Swagger Documentation

✅ Modular & Production-Oriented Design

---

## 🏗 System Architecture

```text
Business Metrics
       │
       ▼
Forecasting Engine
(Moving Average + Trend Analysis)
       │
       ▼
Anomaly Detection Layer
(Z-Score + CUSUM Monitoring)
       │
       ▼
Risk Scoring Engine
       │
       ▼
Incident Classification
       │
       ▼
Action Recommendation API
       │
       ▼
Operational Intelligence Dashboard
```

---

## ⚡ Core Capabilities

### 📈 Forecasting Engine

Generates future projections from historical time-series observations using:

- Moving Average Forecasting
- Trend-Based Forecasting
- Forecast Error Evaluation
- Stability Monitoring

---

### 🚨 Anomaly Detection

Detects abnormal operational patterns using:

- Statistical Z-Score Analysis
- CUSUM Change Detection
- Severity Scoring
- Risk Categorization

Example detections:

- Revenue spikes
- Service degradation
- System instability
- Operational anomalies

---

### 🎯 Risk Intelligence

Converts monitoring signals into business actions.

Outputs:

- Incident Risk Score
- Alert Priority
- Recommended Action
- Downtime Risk Estimation
- Operational Impact Assessment

---

## 🔧 Technology Stack

### Backend

- Python
- FastAPI
- Pydantic
- Uvicorn

### Analytics

- NumPy
- Statistical Forecasting
- Z-Score Detection
- CUSUM Monitoring

### DevOps

- Docker
- Git
- GitHub

### API Documentation

- OpenAPI
- Swagger UI

---

## 📊 Sample API Output

```json
{
  "risk_level": "high",
  "incident_risk_score": 78.88,
  "anomalies_detected": 1,
  "cusum_signal": true,
  "recommended_action": "Investigate immediately and trigger incident response workflow"
}
```

---

## 📡 API Endpoints

### Health Check

```http
GET /health
```

Returns service health status.

---

### Analyze Time Series

```http
POST /analyze
```

Input:

```json
{
  "values": [120,122,121,125,128,130,127,400,132,129,131,135],
  "forecast_horizon": 5
}
```

Returns:

- Forecast Results
- Anomaly Detection Report
- Risk Assessment
- Recommended Action

---

### Monitoring Metrics

```http
GET /metrics
```

Provides operational monitoring statistics and system KPIs.

---

## 📈 Business Value

Sentinel demonstrates how modern AI-assisted monitoring systems can:

- Detect anomalies before incidents escalate
- Reduce manual monitoring effort
- Improve operational visibility
- Prioritize incident response
- Generate actionable intelligence from enterprise-scale time-series data

---

## 🖥 Demo

### Swagger Documentation

Available at:

```text
http://127.0.0.1:8020/docs
```

---

## 🚀 Run Locally

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8020
```

Open:

```text
http://127.0.0.1:8020/docs
```

---

## 📌 Project Impact

This project demonstrates expertise in:

- Backend Engineering
- API Design
- Monitoring Systems
- Forecasting Pipelines
- Anomaly Detection
- Risk Intelligence Platforms
- Production-Oriented Python Development

---

## 👨‍💻 Author

**Vansh Patel**

MS Computer Science (AI/ML)
Stevens Institute of Technology

Interested in:
- AI Engineering
- Forward Deployed Engineering
- Machine Learning Systems
- Decision Intelligence Platforms
- Production AI Infrastructure