import os
import csv
import random
import numpy as np
from PIL import Image
import cv2


def opencv_preprocessing(pil_image):
    image = np.array(pil_image)
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=10, templateWindowSize=5, searchWindowSize=19
    )
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(equalized_rgb)


def random_resized_crop(pil_image, size=512, scale_min=0.3, scale_max=1.0):
    w, h = pil_image.size
    area = w * h
    target_area = area * random.uniform(scale_min, scale_max)
    aspect_ratio = random.uniform(3.0 / 4.0, 4.0 / 3.0)
    crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
    crop_h = int(round(np.sqrt(target_area / aspect_ratio)))
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    x = random.randint(0, max(0, w - crop_w))
    y = random.randint(0, max(0, h - crop_h))
    cropped = pil_image.crop((x, y, x + crop_w, y + crop_h))
    return cropped.resize((size, size), Image.BILINEAR)


def random_rotation(pil_image, degrees=50):
    angle = random.uniform(-degrees, degrees)
    return pil_image.rotate(angle, resample=Image.BILINEAR)


def random_horizontal_flip(pil_image, p=0.5):
    if random.random() < p:
        return pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    return pil_image


def random_vertical_flip(pil_image, p=0.5):
    if random.random() < p:
        return pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    return pil_image


def color_jitter(pil_image, brightness=0.2, contrast=0.3, saturation=0.1):
    img = np.array(pil_image).astype(np.float32)
    if random.random() < 0.8:
        b = 1.0 + random.uniform(-brightness, brightness)
        img = np.clip(img * b, 0, 255)
    if random.random() < 0.8:
        c = 1.0 + random.uniform(-contrast, contrast)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip(mean + (img - mean) * c, 0, 255)
    if random.random() < 0.8:
        s = 1.0 + random.uniform(-saturation, saturation)
        gray = np.mean(img, axis=2, keepdims=True)
        img = np.clip(gray + (img - gray) * s, 0, 255)
    return Image.fromarray(img.astype(np.uint8))


def full_pipeline(pil_image):
    img = opencv_preprocessing(pil_image)
    img = random_resized_crop(img, 512, 0.3, 1.0)
    img = random_rotation(img, 50)
    img = random_horizontal_flip(img)
    img = random_vertical_flip(img)
    img = color_jitter(img)
    return img


def main():
    out_dir = 'preview_output'
    os.makedirs(out_dir, exist_ok=True)

    with open('dataset.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    random.seed(42)
    np.random.seed(42)
    samples = random.sample(rows, min(6, len(rows)))

    for i, row in enumerate(samples):
        path = row['img_name']
        label = row['labels']

        if not os.path.exists(path):
            print(f"[SKIP] Nao encontrado: {path}")
            continue

        original = Image.open(path).convert('RGB')
        w, h = original.size

        # Reduz preview para no max 1200px (denoise e muito lento em 6K)
        if max(w, h) > 1200:
            scale = 1200 / max(w, h)
            preview = original.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            preview = original

        print(f"\n--- {i+1}/6 - {label} ({preview.size[0]}x{preview.size[1]}, original era {w}x{h}) ---")
        print(f"  ...{path[-60:]}")

        preview.save(f'{out_dir}/{i:02d}_{label}_original_{preview.size[0]}x{preview.size[1]}.jpg', quality=95)
        print(f"  [original]")

        denoised = opencv_preprocessing(preview)
        denoised.save(f'{out_dir}/{i:02d}_{label}_denoised.jpg', quality=95)
        print(f"  [denoised + equalizado]")

        for v in range(3):
            aug = full_pipeline(preview)
            aug.save(f'{out_dir}/{i:02d}_{label}_aug_v{v}.jpg', quality=95)
            print(f"  [augment v{v}] (512x512)")

    print(f"\n=== Concluido! Pasta: {os.path.abspath(out_dir)}/ ===")


if __name__ == '__main__':
    main()
