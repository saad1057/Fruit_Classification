import os
import shutil

def merge_fruit_image_folders(root_path="../dataset", fruits=None):
    if fruits is None:
        fruits = [
            "apple",
            "banana",
            "orange",
            "strawberry",
            "mango",
            "peach",
            "grapes",
            "pineapple",
            "watermelon",
            "kiwi"
        ]

    for fruit in fruits:
        src1 = os.path.join(root_path, f"{fruit}_fruit")
        src2 = os.path.join(root_path, f"{fruit}_fruit_images")
        target = os.path.join(root_path, "merged", fruit)
        os.makedirs(target, exist_ok=True)

        counter = 0
        for folder in [src1, src2]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        new_filename = f"{counter:05d}.jpg"
                        dest_path = os.path.join(target, new_filename)
                        shutil.copy2(file_path, dest_path)
                        counter += 1

        print(f"✅ Merged {counter} images for: {fruit}")

# Run the function
merge_fruit_image_folders()
