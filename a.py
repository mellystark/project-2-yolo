import os
import random
import shutil

# Ana dataset yolu
BASE_PATH = r"dataset"

IMAGES_PATH = os.path.join(BASE_PATH, "images")
LABELS_PATH = os.path.join(BASE_PATH, "labels")

TRAIN_RATIO = 0.8  # %80 train, %20 val
EXTENSIONS = (".jpg", ".jpeg", ".png")

# train / val klasörlerini oluştur
for split in ["train", "val"]:
    os.makedirs(os.path.join(IMAGES_PATH, split), exist_ok=True)
    os.makedirs(os.path.join(LABELS_PATH, split), exist_ok=True)

# images içindeki görselleri al (alt klasörler hariç)
images = [
    f for f in os.listdir(IMAGES_PATH)
    if f.lower().endswith(EXTENSIONS)
    and os.path.isfile(os.path.join(IMAGES_PATH, f))
]

random.shuffle(images)

split_index = int(len(images) * TRAIN_RATIO)
train_images = images[:split_index]
val_images = images[split_index:]

def move_files(image_list, split_name):
    for img in image_list:
        name, _ = os.path.splitext(img)

        img_src = os.path.join(IMAGES_PATH, img)
        img_dst = os.path.join(IMAGES_PATH, split_name, img)

        lbl_src = os.path.join(LABELS_PATH, f"{name}.txt")
        lbl_dst = os.path.join(LABELS_PATH, split_name, f"{name}.txt")

        # image taşı
        shutil.move(img_src, img_dst)

        # label varsa taşı (yoksa hata vermez)
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)

# Taşıma işlemleri
move_files(train_images, "train")
move_files(val_images, "val")

print("✔ images ve labels klasörleri train / val olarak birebir eşleşmeli şekilde bölündü.")
