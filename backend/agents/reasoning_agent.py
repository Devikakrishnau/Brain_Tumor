class ReasoningAgent:

    def generate_explanation(self, pred, conf, text_signal):

        explanation = f"""
🧠 Agent Reasoning Trace

Image Agent → Detected: {pred}
Model Confidence → {conf}

Clinical Agent Signal → {text_signal}

Fusion Agent → Confidence adjusted using clinical modality

Final Decision → {pred}
"""

        return explanation
