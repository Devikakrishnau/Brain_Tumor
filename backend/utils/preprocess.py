import cv2
from torchvision import transforms


# ----------------------------------
# Transform — MUST match training
# ----------------------------------

transform_pipeline = transforms.Compose([
    transforms.ToTensor()
])


# ----------------------------------
# Load + preprocess MRI image
# ----------------------------------

def load_image(image_path):
    """
    Reads MRI image and converts to model-ready tensor
    Used by ImageAgent
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"❌ Cannot read image: {image_path}")

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to model input size
    img = cv2.resize(img, (224, 224))

    tensor = transform_pipeline(img)

    # add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor
