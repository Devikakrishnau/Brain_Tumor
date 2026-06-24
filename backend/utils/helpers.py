import os
import uuid


# ✅ save uploaded file safely
def save_temp_file(upload_bytes, suffix=".jpg"):
    name = f"temp_{uuid.uuid4().hex}{suffix}"

    with open(name, "wb") as f:
        f.write(upload_bytes)

    return name


# ✅ format confidence nicely
def format_confidence(conf):
    return round(float(conf) * 100, 2)


# ✅ dataset class labels (single source of truth)
CLASS_LABELS = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]


def get_class_name(index):
    return CLASS_LABELS[index]


# ✅ cleanup temp files
def cleanup_file(path):
    if os.path.exists(path):
        os.remove(path)
