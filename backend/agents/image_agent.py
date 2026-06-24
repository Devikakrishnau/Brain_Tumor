from backend.utils.preprocess import load_image



class ImageAgent:

    def process(self, image_path):
        """
        Converts MRI image → model tensor
        """
        tensor = load_image(image_path)
        return tensor
