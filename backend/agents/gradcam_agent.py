import torch
import cv2
import numpy as np


class GradCAMAgent:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Correct hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, image_path):

        self.model.eval()

        output = self.model(tensor)
        class_idx = torch.argmax(output, dim=1)

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global Average Pooling
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        cam = torch.relu(cam)

        cam = cam.cpu().numpy()

        # Normalize safely
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        cam = cv2.resize(cam, (224, 224))
        self.cam = cam

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        # Read original image correctly
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))

        # Convert BGR → RGB for better alignment
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        out_path = "gradcam_output.jpg"

        # Convert back to BGR before saving
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_path, overlay)

        return out_path