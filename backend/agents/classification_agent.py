import torch
from backend.utils.helpers import CLASS_LABELS


class ClassificationAgent:

    def predict(self, model, tensor):
        """
        Runs model inference and returns
        class name + confidence
        """

        model.eval()

        with torch.no_grad():
            output = model(tensor)

        probs = torch.softmax(output, dim=1)
        conf, idx = torch.max(probs, dim=1)

        class_name = CLASS_LABELS[idx.item()]
        confidence = conf.item()

        return class_name, confidence
