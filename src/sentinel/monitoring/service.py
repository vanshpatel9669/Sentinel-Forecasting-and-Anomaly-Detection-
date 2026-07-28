"""Turns forecast error + anomaly severity into an actionable incident-risk
score. `calculate_incident_risk` is a real, deterministic function of its
inputs — a rule-based scorer, not a placeholder for a model that was never
built.

The previous version of this module also returned a second block,
`estimate_operational_impact`, containing two hardcoded literal strings —
`"response_time_reduction": "35%"` and `"forecast_stability_improvement":
"22%"` — on every single response, regardless of input. There is no
honest way to compute a real incident-response-time or downtime reduction
percentage from a single request against a demo service with no
production deployment to compare against, so that block is removed here
rather than replaced with a different fabricated number. Real, genuinely
computed operational monitoring instead comes from Prometheus counters
(`api/routes.py`'s `analyses_total` / `alerts_total`) exposed at
`GET /metrics` — see docs/architecture.md for the full account.
"""

from __future__ import annotations


def calculate_incident_risk(
    severity_score: float, forecast_rmse: float, anomaly_count: int
) -> dict[str, object]:
    risk_score = 0.0
    risk_score += severity_score * 0.5
    risk_score += min(forecast_rmse * 2, 25)
    risk_score += min(anomaly_count * 10, 25)
    risk_score = min(100, round(risk_score, 2))

    if risk_score >= 70:
        priority = "high"
        recommended_action = "Investigate immediately and trigger incident response workflow"
    elif risk_score >= 40:
        priority = "medium"
        recommended_action = "Monitor closely and review recent operational changes"
    else:
        priority = "low"
        recommended_action = "Continue normal monitoring"

    return {
        "incident_risk_score": risk_score,
        "alert_priority": priority,
        "recommended_action": recommended_action,
    }
