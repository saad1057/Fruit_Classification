from PIL import Image
import os

def validate_images(folder):
    total, removed = 0, 0
    for cls in os.listdir(folder):
        class_path = os.path.join(folder, cls)
        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            total += 1
            try:
                img = Image.open(img_path)
                img.verify()
            except:
                os.remove(img_path)
                removed += 1
                print(f"Removed corrupted: {img_path}")
    print(f"\nCleaned: {removed}/{total} invalid images removed.")

validate_images("../dataset")
