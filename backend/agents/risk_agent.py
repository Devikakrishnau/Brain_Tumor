class RiskAgent:

    def evaluate(self, prediction, confidence, tumor_size_cm, clinical_text):

        text = (clinical_text or "").lower()

        # ------------------------------------------------
        # If no tumor detected → risk = LOW
        # ------------------------------------------------
        if prediction.lower() == "no_tumor":

            return {
                "risk_score": 0.0,
                "risk_level": "Low"
            }

        # ------------------------------------------------
        # Tumor Size Score
        # ------------------------------------------------
        if tumor_size_cm < 2:
            size_score = 1
        elif tumor_size_cm < 4:
            size_score = 2
        else:
            size_score = 3

        # ------------------------------------------------
        # Symptom Score
        # ------------------------------------------------
        symptom_score = 0

        severe = ["seizure", "weakness", "unconscious"]
        moderate = ["vision", "blur"]
        mild = ["headache", "dizziness"]

        if any(w in text for w in severe):
            symptom_score = 3
        elif any(w in text for w in moderate):
            symptom_score = 2
        elif any(w in text for w in mild):
            symptom_score = 1
        else:
            symptom_score = 0

        # ------------------------------------------------
        # Tumor Type Score
        # ------------------------------------------------
        tumor = prediction.lower()

        if "glioma" in tumor:
            tumor_score = 3
        elif "pituitary" in tumor:
            tumor_score = 2
        elif "meningioma" in tumor:
            tumor_score = 1
        else:
            tumor_score = 1

        # ------------------------------------------------
        # Confidence uncertainty
        # ------------------------------------------------
        if confidence < 0.6:
            uncertainty_score = 2
        elif confidence < 0.8:
            uncertainty_score = 1
        else:
            uncertainty_score = 0

        # ------------------------------------------------
        # Final Score
        # ------------------------------------------------
        total = size_score + symptom_score + tumor_score + uncertainty_score

        if total >= 6:
            risk_level = "High"
        elif total >= 3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        risk_score = total / 10

        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level
        }