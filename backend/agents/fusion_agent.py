class FusionAgent:

    def fuse(self, image_pred, confidence, text_signal):

        adjusted_conf = confidence

        # agent-based confidence adjustment
        if text_signal == "high_risk":
            adjusted_conf += 0.15

        elif text_signal == "medium_risk":
            adjusted_conf += 0.07

        adjusted_conf = min(adjusted_conf, 0.99)

        return image_pred, round(adjusted_conf, 3)
