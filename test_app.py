import os
import cv2
from ultralytics import YOLO

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QListWidget,
    QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QGroupBox
)

CONFIDENCE = 0.25  # Sabit eşik (slider yok)


class YoloGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Object Detection - BLG-407")
        self.setMinimumSize(1100, 650)

        # ================= STATE =================
        self.model = None
        self.model_path = None

        self.image_path = None
        self.original_bgr = None
        self.tagged_bgr = None

        # ================= UI =================
        self._build_ui()
        self.load_model()  # best.pt otomatik yükleme

    # ================= UI BUILDER =================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout()
        central.setLayout(root_layout)

        title = QLabel("YOLOv8 Detection GUI (PyQt5)")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)

        main_layout = QHBoxLayout()
        root_layout.addLayout(main_layout)

        # ---------- Original Image Panel ----------
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout()
        original_group.setLayout(original_layout)

        self.original_label = QLabel("No image selected")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet(
            "background-color:#eeeeee; border:1px solid #b0b0b0;"
        )
        self.original_label.setMinimumSize(500, 400)
        original_layout.addWidget(self.original_label)

        main_layout.addWidget(original_group)

        # ---------- Tagged Image Panel ----------
        tagged_group = QGroupBox("Tagged Image")
        tagged_layout = QVBoxLayout()
        tagged_group.setLayout(tagged_layout)

        self.tagged_label = QLabel("No inference yet")
        self.tagged_label.setAlignment(Qt.AlignCenter)
        self.tagged_label.setStyleSheet(
            "background-color:#eeeeee; border:1px solid #b0b0b0;"
        )
        self.tagged_label.setMinimumSize(500, 400)
        tagged_layout.addWidget(self.tagged_label)

        # Stats
        self.count_label = QLabel("Detected objects: -")
        self.count_label.setFont(QFont("Arial", 11, QFont.Bold))
        tagged_layout.addWidget(self.count_label)

        self.class_list = QListWidget()
        tagged_layout.addWidget(self.class_list)

        main_layout.addWidget(tagged_group)

        # ---------- Buttons ----------
        btn_layout = QHBoxLayout()
        root_layout.addLayout(btn_layout)

        self.btn_select = QPushButton("Select Image")
        self.btn_select.clicked.connect(self.select_image)
        btn_layout.addWidget(self.btn_select)

        self.btn_test = QPushButton("Test Image")
        self.btn_test.clicked.connect(self.test_image)
        btn_layout.addWidget(self.btn_test)

        self.btn_save = QPushButton("Save Image")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)  # inference sonrası aktif
        btn_layout.addWidget(self.btn_save)

    # ================= MODEL LOAD =================
    def load_model(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best.pt")

        if not os.path.exists(model_path):
            QMessageBox.critical(
                self,
                "Model not found",
                "best.pt was not found in the same folder as gui_app.py.\n"
                "Please place best.pt next to gui_app.py."
            )
            return

        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
        except Exception as e:
            QMessageBox.critical(self, "Model load error", str(e))
            self.model = None

    # ================= SELECT IMAGE =================
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "Error", "Image could not be read.")
            return

        self.image_path = file_path
        self.original_bgr = img
        self.tagged_bgr = None

        self.render_to_label(self.original_bgr, self.original_label)
        self.tagged_label.setText("No inference yet")
        self.count_label.setText("Detected objects: -")
        self.class_list.clear()
        self.btn_save.setEnabled(False)

    # ================= TEST IMAGE =================
    def test_image(self):
        if self.model is None:
            QMessageBox.warning(self, "Model error", "Model is not loaded.")
            return

        if self.original_bgr is None:
            QMessageBox.warning(self, "No image", "Please select an image first.")
            return

        try:
            results = self.model.predict(
                self.original_bgr,
                conf=CONFIDENCE,
                verbose=False
            )
        except Exception as e:
            QMessageBox.critical(self, "Inference error", str(e))
            return

        tagged, detections = self.annotate(self.original_bgr.copy(), results)
        self.tagged_bgr = tagged

        self.render_to_label(self.tagged_bgr, self.tagged_label)
        self.update_stats(detections)
        self.btn_save.setEnabled(True)

    # ================= ANNOTATE =================
    def annotate(self, img_bgr, results):
        r = results[0]
        boxes = r.boxes
        detections = []

        names = self._get_class_names()

        if boxes is None or len(boxes) == 0:
            cv2.putText(
                img_bgr, "NO DETECTION", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )
            return img_bgr, detections

        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])

            class_name = names[cls_id]
            detections.append(class_name)

            color = self._class_color(class_name)
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img_bgr, label, (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

        return img_bgr, detections

    # ================= UPDATE STATS =================
    def update_stats(self, detections):
        total = len(detections)
        self.count_label.setText(f"Detected objects: {total}")
        self.class_list.clear()

        if total == 0:
            self.class_list.addItem("-")
            return

        counts = {}
        for c in detections:
            counts[c] = counts.get(c, 0) + 1

        for c, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            self.class_list.addItem(f"{c}  x{n}")

    # ================= SAVE IMAGE =================
    def save_image(self):
        if self.tagged_bgr is None:
            QMessageBox.warning(self, "Nothing to save", "Please run Test Image first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Tagged Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if not save_path:
            return

        try:
            ok = cv2.imwrite(save_path, self.tagged_bgr)
            if not ok:
                raise IOError("cv2.imwrite failed")

            QMessageBox.information(
                self,
                "Saved",
                f"Tagged image saved successfully:\n{save_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    # ================= RENDER =================
    def render_to_label(self, img_bgr, label: QLabel):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg)

        label.setPixmap(
            pix.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # ================= HELPERS =================
    def _get_class_names(self):
        names = self.model.names
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    def _class_color(self, name):
        h = abs(hash(name))
        return (h % 256, (h // 256) % 256, (h // 65536) % 256)


# ================= MAIN =================
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = YoloGui()
    win.show()
    sys.exit(app.exec_())
