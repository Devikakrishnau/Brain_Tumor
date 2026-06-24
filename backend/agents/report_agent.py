class ReportAgent:

    def generate(self, prediction, confidence, risk_level):

        return f"""
AI RADIOLOGY SUMMARY
----------------------------------

Tumor Type      : {prediction}
Model Confidence: {confidence*100:.2f} %
Clinical Risk   : {risk_level}

----------------------------------
"""