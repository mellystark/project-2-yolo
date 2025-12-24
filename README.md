# YOLOv8 ile Nesne Tespiti ve PyQt5 Tabanlı GUI Uygulaması

Bu proje, BLG-407 Makine Öğrenmesi dersi kapsamında YOLOv8 tabanlı bir nesne tespit sistemi geliştirmek amacıyla hazırlanmıştır.  
Çalışmada, özel olarak oluşturulmuş bir veri seti üzerinde eğitilen YOLOv8 modeli kullanılarak nesne tespiti yapılmış ve sonuçlar PyQt5 tabanlı bir masaüstü arayüz üzerinden görselleştirilmiştir.

Proje; model eğitimi, inference süreci ve kullanıcı etkileşimli bir GUI uygulamasını bütüncül şekilde içermektedir.

---

## 2️⃣ Proje Kapsamı

Bu projede aşağıdaki adımlar gerçekleştirilmiştir:

- YOLOv8 mimarisi kullanılarak nesne tespit modeli eğitilmiştir
- Öğrenci tarafından oluşturulan özel veri seti YOLO formatında etiketlenmiştir
- Eğitilen modelin sonuçları PyQt5 tabanlı bir GUI ile görselleştirilmiştir
- Kullanıcıya inference sonucu görüntüyü kaydetme imkânı sunulmuştur

Kullanılan teknolojiler:
- **YOLOv8**
- **PyQt5**
- **OpenCV**
- **Python**

---

## 3️⃣ Veri Seti Bilgisi

- Veri seti öğrenci tarafından manuel olarak oluşturulmuştur
- En az **2 sınıf** ve **200’den fazla görüntü** içermektedir
- Tüm etiketler **YOLO formatında** hazırlanmıştır
- Sınıf örnekleri:
  - `battery`
  - `flash`
- Veri seti eğitim ve doğrulama (train/val) olacak şekilde ayrılmıştır

---

## 4️⃣ Model Eğitimi

- Model eğitimi **YOLOv8** kullanılarak gerçekleştirilmiştir
- Eğitim süreci ve tüm deneyler `yolo_training.ipynb` dosyasında yer almaktadır
- Eğitim sonunda aşağıdaki çıktılar elde edilmiştir:
  - mAP metrikleri
  - Loss grafikleri
  - En iyi modeli temsil eden **best.pt** dosyası
- Eğitim sonrası inference testleri notebook içerisinde gösterilmiştir

---

## 5️⃣ PyQt5 GUI Uygulaması

Bu proje kapsamında, eğitilen YOLOv8 modelini test etmek amacıyla PyQt5 tabanlı bir masaüstü uygulaması geliştirilmiştir.

GUI uygulamasının özellikleri:

- **Original Image Paneli**
  - Seçilen ham görüntünün gösterimi
- **Tagged Image Paneli**
  - YOLOv8 tarafından tespit edilen nesnelerin bounding box’lar ile gösterimi
- Arayüz fonksiyonları:
  - Görüntü seçme
  - Nesne tespiti (inference)
  - Bounding box çizimi
  - Tespit edilen nesne sayısı ve sınıf listesinin gösterimi
  - Sonuç görüntüsünü diske kaydetme (**Save Image**)

GUI uygulaması `gui_app.py` dosyasında yer almaktadır.

---

## 6️⃣ Proje Klasör Yapısı

```text
project-2-yolo/
│
├── yolo_training.ipynb
├── gui_app.py
├── best.pt
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── README.md
└── requirements.txt
````

---

## 7️⃣ Kurulum ve Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

GUI uygulamasını çalıştırmak için:

```bash
python gui_app.py
```

> **Not:** `best.pt` dosyası `gui_app.py` ile aynı dizinde bulunmalıdır.

---

## 8️⃣ Olası Hatalar ve Çözümler

* **Model yüklenmiyor:**
  `best.pt` dosyasının doğru dizinde olduğundan emin olun.

* **OpenCV veya PyQt5 hatası:**
  `requirements.txt` dosyasındaki kütüphanelerin eksiksiz kurulduğunu kontrol edin.

* **Görüntü yüklenmiyor:**
  Seçilen dosyanın desteklenen bir görüntü formatı olduğundan emin olun (`jpg`, `png`, `jpeg`).

---

Bu proje, BLG-407 Makine Öğrenmesi dersi kapsamında istenen tüm teknik ve görsel gereksinimleri karşılayacak şekilde hazırlanmıştır.

```
```
