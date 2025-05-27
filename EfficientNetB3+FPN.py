# SSD-based Object Detection Model with EfficientNetB0 Backbone
# Dataset: Pascal VOC 2007 (Multi-class Detection with 5-Fold Validation)

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB0
from xml.etree import ElementTree as ET
from tqdm import tqdm
import urllib.request
import random
import tarfile
import matplotlib.pyplot as plt
from tensorflow.image import non_max_suppression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import Callback

# === Pascal VOC Classes ===
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
class_map = {cls: i+1 for i, cls in enumerate(VOC_CLASSES)}  # Background=0

# === Download Pascal VOC 2007 ===
def download_and_extract_voc2007(dest_dir='VOCdevkit'):
    voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    test_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    os.makedirs(dest_dir, exist_ok=True)
    for url in [voc_url, test_url]:
        filename = url.split('/')[-1]
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        print(f"Extracting {filename}...")
        with tarfile.open(filepath) as tar:
            tar.extractall(path=dest_dir)

# === Locate Image and Annotation Paths ===
def find_voc_paths(base_dir="VOCdevkit"):
    for root, dirs, files in os.walk(base_dir):
        if "JPEGImages" in dirs and "Annotations" in dirs:
            return os.path.join(root, "JPEGImages"), os.path.join(root, "Annotations")
    raise FileNotFoundError("VOC JPEGImages or Annotations folder not found.")

# === ID List Utility ===
def get_or_create_id_list(txt_path, jpeg_dir):
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return f.read().strip().split()
    ids = [fname.split('.')[0] for fname in os.listdir(jpeg_dir) if fname.endswith('.jpg')]
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(ids))
    return ids

# === Anchor Box Generator ===
def generate_anchors(feature_map_sizes, image_size=224, scales=[0.2, 0.4, 0.6], ratios=[0.5, 1.0, 2.0]):
    anchors = []
    for idx, fmap_size in enumerate(feature_map_sizes):
        scale = scales[idx]
        step = image_size / fmap_size
        for i in range(fmap_size):
            for j in range(fmap_size):
                cy = (i + 0.5) * step / image_size
                cx = (j + 0.5) * step / image_size
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    anchors.append([cx, cy, w, h])
    return np.clip(np.array(anchors), 0.0, 1.0)

# === IOU Calculation ===
def compute_iou(box, anchors):
    box = np.expand_dims(box, axis=0)
    box_xy = box[:, :2]
    box_wh = box[:, 2:]
    box_min = box_xy - box_wh / 2
    box_max = box_xy + box_wh / 2

    anchors_xy = anchors[:, :2]
    anchors_wh = anchors[:, 2:]
    anchors_min = anchors_xy - anchors_wh / 2
    anchors_max = anchors_xy + anchors_wh / 2

    intersect_min = np.maximum(box_min, anchors_min)
    intersect_max = np.minimum(box_max, anchors_max)
    intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
    intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]

    box_area = box_wh[:, 0] * box_wh[:, 1]
    anchors_area = anchors_wh[:, 0] * anchors_wh[:, 1]

    union_area = box_area + anchors_area - intersect_area
    return intersect_area / union_area

# === Match Anchors to GT Boxes ===
def match_anchors_to_gt(gt_boxes, gt_labels, anchors, iou_threshold=0.5):
    encoded_boxes = np.zeros_like(anchors)
    labels = np.zeros((anchors.shape[0],), dtype=np.int32)
    for gt, cls_id in zip(gt_boxes, gt_labels):
        ious = compute_iou(gt, anchors)
        best_anchor = np.argmax(ious)
        if ious[best_anchor] > iou_threshold:
            labels[best_anchor] = cls_id
            encoded_boxes[best_anchor] = encode_box(gt, anchors[best_anchor])
    return encoded_boxes, labels

# === Encode Boxes for Regression ===
def encode_box(gt_box, anchor):
    dx, dy, dw, dh = gt_box
    acx, acy, aw, ah = anchor
    dx = (dx - acx) / aw
    dy = (dy - acy) / ah
    dw = np.log(dw / aw)
    dh = np.log(dh / ah)
    return np.array([dx, dy, dw, dh])

# === Parse Pascal VOC Annotations ===
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [] , []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in class_map:
            continue
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        cx = (x_min + x_max) / 2 / 224
        cy = (y_min + y_max) / 2 / 224
        w = (x_max - x_min) / 224
        h = (y_max - y_min) / 224
        boxes.append([cx, cy, w, h])
        labels.append(class_map[name])
    return boxes, labels

# === Focal Loss Function (γ=1.0, α=0.1) ===
def focal_loss(gamma=1.0, alpha=0.1):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)  # shape: (batch, anchors)
        y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])  # convert to one-hot
        
        ce = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
        probs = tf.reduce_sum(y_pred * y_true_onehot, axis=-1)
        focal_weight = tf.pow(1.0 - probs, gamma)
        alpha_weight = tf.where(tf.equal(y_true, 0), 1. - alpha, alpha)
        return tf.reduce_mean(focal_weight * alpha_weight * ce)
    return loss_fn

# === Learning Rate Scheduler ===
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=2000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# === SSD Model ===
def ssd_model(input_shape=(224, 224, 3), num_anchors=27, num_classes=21):
    base = EfficientNetB3(include_top=False, input_shape=input_shape, weights='imagenet')
    x = base.output
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    bbox_head = layers.Dense(num_anchors * 4)(x)
    bbox_head = layers.Reshape((num_anchors, 4), name="bbox")(bbox_head)

    cls_head = layers.Dense(num_anchors * num_classes)(x)
    cls_head = layers.Reshape((num_anchors, num_classes), name="cls")(cls_head)

    return models.Model(inputs=base.input, outputs=[bbox_head, cls_head])

# === Load and Prepare Data ===
def load_data(image_dir, annotation_dir, ids, anchors):
    images, encoded_boxes, encoded_labels = [], [], []
    for id in tqdm(ids):
        img_path = os.path.join(image_dir, id + '.jpg')
        xml_path = os.path.join(annotation_dir, id + '.xml')
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224)) / 255.0
        boxes, labels = parse_voc_annotation(xml_path)
        eb, el = match_anchors_to_gt(boxes, labels, anchors)
        images.append(img)
        encoded_boxes.append(eb)
        encoded_labels.append(el)
    return np.array(images), np.array(encoded_boxes), np.array(encoded_labels)

# === Decode Box Predictions ===
def decode_box(encoded, anchor):
    dx, dy, dw, dh = encoded
    acx, acy, aw, ah = anchor
    cx = dx * aw + acx
    cy = dy * ah + acy
    w = np.exp(dw) * aw
    h = np.exp(dh) * ah
    return [cx, cy, w, h]


# === Draw Predicted and Ground Truth Boxes on Image (NMS + higher threshold + sharpened logits) ===
def draw_boxes(img, pred_boxes, pred_logits, gt_boxes, gt_labels, anchors, threshold=0.75, iou_thresh=0.3):
    img = (img * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]

    boxes_xyxy = []
    scores = []
    classes = []

    for i, (logits, pbox) in enumerate(zip(pred_logits, pred_boxes)):
        logits = logits / 0.7  # Temperature scaling to sharpen predictions
        softmax = tf.nn.softmax(logits).numpy()
        plabel = np.argmax(softmax)
        confidence = np.max(softmax)
        if plabel > 0 and confidence > threshold:
            cx, cy, bw, bh = decode_box(pbox, anchors[i])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            boxes_xyxy.append([y1, x1, y2, x2])
            scores.append(confidence)
            classes.append(plabel)

    if boxes_xyxy:
        indices = tf.image.non_max_suppression(
            boxes=tf.convert_to_tensor(boxes_xyxy, dtype=tf.float32),
            scores=tf.convert_to_tensor(scores, dtype=tf.float32),
            max_output_size=20,
            iou_threshold=iou_thresh
        )
        for idx in indices:
            idx = int(idx.numpy())
            y1, x1, y2, x2 = boxes_xyxy[idx]
            plabel = classes[idx]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, VOC_CLASSES[plabel - 1], (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    for i, gtlabel in enumerate(gt_labels):
        if gtlabel > 0:
            cx, cy, bw, bh = decode_box(gt_boxes[i], anchors[i])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, VOC_CLASSES[gtlabel - 1], (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    return img

# === Match Anchors to GT Boxes with Better Logic ===
def match_anchors_to_gt(gt_boxes, gt_labels, anchors, iou_threshold=0.5):
    encoded_boxes = np.zeros_like(anchors)
    labels = np.zeros((anchors.shape[0],), dtype=np.int32)

    for gt, cls_id in zip(gt_boxes, gt_labels):
        ious = compute_iou(gt, anchors)
        best_anchor = np.argmax(ious)
        labels[best_anchor] = cls_id
        encoded_boxes[best_anchor] = encode_box(gt, anchors[best_anchor])

        positive_indices = np.where(ious > iou_threshold)[0]
        for idx in positive_indices:
            labels[idx] = cls_id
            encoded_boxes[idx] = encode_box(gt, anchors[idx])

    return encoded_boxes, labels

# === Main Execution with 5-Fold Cross Validation ===
if __name__ == '__main__':
    download_and_extract_voc2007()
    image_dir, annotation_dir = find_voc_paths()
    ids = get_or_create_id_list("VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", image_dir)
    anchors = generate_anchors([7, 4, 2])

    images, boxes, labels = load_data(image_dir, annotation_dir, ids, anchors)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat_labels = np.array([np.max(l) for l in labels])

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, strat_labels), 1):
        print(f"\n--- Fold {fold} ---")
        x_train, x_val = images[train_idx], images[val_idx]
        y_train_box, y_val_box = boxes[train_idx], boxes[val_idx]
        y_train_cls, y_val_cls = labels[train_idx], labels[val_idx]

        model = ssd_model(num_anchors=anchors.shape[0], num_classes=21)
        model.compile(optimizer=optimizer,
                      loss={"bbox": tf.keras.losses.Huber(), "cls": focal_loss(gamma=1.0, alpha=0.1)},
                      metrics={"cls": "accuracy"})

        history = model.fit(
            x_train,
            {"bbox": y_train_box, "cls": y_train_cls},
            validation_data=(x_val, {"bbox": y_val_box, "cls": y_val_cls}),
            epochs=25,
            batch_size=8
        )

        preds = model.predict(x_val)
        y_true_flat = []
        y_pred_flat = []
        for i in range(len(x_val)):
            true_labels = y_val_cls[i]
            pred_logits = preds[1][i]
            pred_labels = np.argmax(pred_logits, axis=-1)
            confidences = np.max(pred_logits, axis=-1)
            for anchor_idx in range(len(anchors)):
                if confidences[anchor_idx] > 0.5:
                    y_true_flat.append(true_labels[anchor_idx])
                    y_pred_flat.append(pred_labels[anchor_idx])

        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(21))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BG"] + VOC_CLASSES)
        disp.plot(xticks_rotation=90, cmap='Blues')
        plt.title(f"Confusion Matrix - Fold {fold} (Filtered by Confidence > 0.5)")
        plt.show()

        # Visualize 5 Random Predictions
        indices = random.sample(range(len(x_val)), 5)
        for idx in indices:
            img = x_val[idx]
            pred_box = preds[0][idx]
            pred_logits = preds[1][idx]
            gt_box = y_val_box[idx]
            gt_cls = y_val_cls[idx]
            vis_img = draw_boxes(img, pred_box, pred_logits, gt_box, gt_cls, anchors, threshold=0.5)
            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Prediction vs Ground Truth - Fold {fold}")
            plt.axis('off')
            plt.show()
