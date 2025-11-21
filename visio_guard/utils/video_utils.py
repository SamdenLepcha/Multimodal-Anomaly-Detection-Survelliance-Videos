import os
import uuid
import cv2


def save_uploaded_video(file, upload_dir):
    """ Save uploaded video to static/uploads with a random UUID filename """
    os.makedirs(upload_dir, exist_ok=True)

    ext = file.filename.split(".")[-1]
    fname = f"{uuid.uuid4().hex}.{ext}"

    path = os.path.join(upload_dir, fname)
    file.save(path)

    return path
