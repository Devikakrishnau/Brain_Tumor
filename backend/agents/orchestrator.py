from backend.agents.image_agent import ImageAgent
from backend.agents.classification_agent import ClassificationAgent
from backend.agents.clinical_agent import ClinicalAgent
from backend.agents.fusion_agent import FusionAgent
from backend.agents.reasoning_agent import ReasoningAgent

# feature agents
from backend.agents.gradcam_agent import GradCAMAgent
from backend.agents.risk_agent import RiskAgent
from backend.agents.size_agent import SizeAgent


class Orchestrator:

    def __init__(self, model):

        self.model = model

        # core agents
        self.image_agent = ImageAgent()
        self.class_agent = ClassificationAgent()
        self.clinical_agent = ClinicalAgent()
        self.fusion_agent = FusionAgent()
        self.reason_agent = ReasoningAgent()

        # feature agents
        self.risk_agent = RiskAgent()
        self.size_agent = SizeAgent()

        # GradCAM uses last conv block of resnet
        try:
            target_layer = model.layer4[-1]
            self.gradcam_agent = GradCAMAgent(model, target_layer)
        except:
            self.gradcam_agent = None

    # -------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------
    def run_pipeline(self, image_path, clinical_text):

        # 1️⃣ Image preprocessing
        tensor = self.image_agent.process(image_path)

        # 2️⃣ Classification
        pred, conf = self.class_agent.predict(self.model, tensor)

        # 3️⃣ Clinical text signal
        text_signal = self.clinical_agent.analyze(clinical_text)

        # 4️⃣ Fusion
        final_pred, final_conf = self.fusion_agent.fuse(
            pred, conf, text_signal
        )

        # 5️⃣ Tumor size estimation
        # Tumor size estimation
        if final_pred.lower() == "no_tumor":
            size_cm = 0
        else:
            size_cm = self.size_agent.estimate(image_path)
        # 6️⃣ Risk evaluation (multi-factor)
        risk_result = self.risk_agent.evaluate(
        prediction=final_pred,
        confidence=float(final_conf),
        tumor_size_cm=size_cm,
        clinical_text=clinical_text
       )

        risk_score = risk_result["risk_score"]
        risk_level = risk_result["risk_level"]

        # 7️⃣ GradCAM explainability
        heatmap_path = None
        if self.gradcam_agent is not None:
            try:
                heatmap_path = self.gradcam_agent.generate(
                    tensor,
                    image_path
                )
            except:
                heatmap_path = None

        # 8️⃣ Reasoning explanation
        explanation = self.reason_agent.generate_explanation(
            final_pred,
            final_conf,
            text_signal
        )

        # -------------------------------------------------
        # RETURN OUTPUT (NO REPORT GENERATION HERE)
        # -------------------------------------------------
        return {
            "prediction": final_pred,
            "confidence": float(final_conf),
            "clinical_signal": text_signal,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "tumor_size_cm": size_cm,
            "gradcam_image": heatmap_path,
            "reasoning": explanation
        }