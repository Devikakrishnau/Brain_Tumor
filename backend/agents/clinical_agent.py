class ClinicalAgent:

    def analyze(self, text):
        """
        Simple rule-based clinical signal extractor
        """

        if not text:
            return "no_signal"

        text = text.lower()

        if "seizure" in text or "vision loss" in text:
            return "high_risk"

        if "headache" in text or "vomiting" in text:
            return "medium_risk"

        return "low_risk"
