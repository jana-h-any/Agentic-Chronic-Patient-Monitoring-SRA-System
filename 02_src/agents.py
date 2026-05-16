from utils import log_to_json
from datetime import datetime

class MessageBus:
    def __init__(self):
        self.messages = []

    def publish(self, topic, message):
        msg = {
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'message': message
        }
        self.messages.append(msg)
        return msg

class BaseAgent:
    def __init__(self, name, bus):
        self.name = name
        self.bus = bus

    def log_decision(self, decision):
        log_to_json("logs/agent_decisions.json", {
            "agent": self.name,
            "decision": decision
        })

class VitalsAgent(BaseAgent):
    def analyze(self, patient_data):
        critical = patient_data['prob_critical'] > 0.5
        self.log_decision(f"Vitals analysis: Critical={critical}")
        return self.bus.publish("vitals", {"critical": critical, "prob_critical": float(patient_data['prob_critical'])})

class TrendAgent(BaseAgent):
    def analyze(self, actual_hr, forecast_hr):
        deviation = abs(actual_hr - forecast_hr) / forecast_hr if forecast_hr else 0
        alert = deviation > 0.20
        self.log_decision(f"Trend analysis: deviation={deviation:.2f}, alert={alert}")
        return self.bus.publish("trends", {"alert": alert, "deviation": deviation, "actual_hr": float(actual_hr), "forecast_hr": float(forecast_hr)})

class RiskAgent(BaseAgent):
    def analyze(self, future_risk):
        self.log_decision(f"Risk assessment: risk_score={future_risk:.2f}")
        return self.bus.publish("risk", {"future_risk": float(future_risk)})

class AlertAgent(BaseAgent):
    def process_alerts(self, current_fpr, threshold):
        self.log_decision(f"Alert configuration updated: threshold={threshold:.2f} (FPR={current_fpr:.2f})")
        status = "tightened" if current_fpr > 0.15 else "stable"
        return self.bus.publish(
            "alerts",
            {"current_fpr": float(current_fpr), "threshold": float(threshold), "status": status},
        )
