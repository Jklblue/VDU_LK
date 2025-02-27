from PIL import Image
import os
from vdunet import VDUnet

if __name__ == "__main__":

    name_classes = ["background", "Oilseed_rape"]
    input_folder = "images"
    output_folder = "images_out"

    os.makedirs(output_folder, exist_ok=True)
    model = VDUnet()
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path)
            r_image = model.detect_image(image, count=False, name_classes=name_classes)
            output_path = os.path.join(output_folder, filename.split('.')[0] + 'd.' + filename.split('.')[-1])
            r_image.save(output_path)
            print(f"Processed and saved: {output_path}")

