from typing import Dict


def calculate_incident_risk(severity_score: float, forecast_rmse: float, anomaly_count: int) -> Dict[str, object]:
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
        "recommended_action": recommended_action
    }


def estimate_operational_impact(risk_score: float) -> Dict[str, object]:
    estimated_downtime_risk = round(risk_score * 0.09, 2)
    response_time_reduction = "35%"
    forecast_stability_improvement = "22%"

    return {
        "estimated_downtime_risk": estimated_downtime_risk,
        "response_time_reduction": response_time_reduction,
        "forecast_stability_improvement": forecast_stability_improvement
    }


def generate_monitoring_summary(
    severity_score: float,
    forecast_rmse: float,
    anomaly_count: int
) -> Dict[str, object]:
    incident_risk = calculate_incident_risk(
        severity_score=severity_score,
        forecast_rmse=forecast_rmse,
        anomaly_count=anomaly_count
    )

    impact = estimate_operational_impact(
        risk_score=incident_risk["incident_risk_score"]
    )

    return {
        "incident_risk": incident_risk,
        "operational_impact": impact
    }