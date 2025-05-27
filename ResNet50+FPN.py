# === SSD Model with ResNet50 + FPN, GIoU, Multi-Label Classification, Focal Loss, Anchor Matching, NMS ===
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import os, cv2, tarfile, urllib.request
from xml.etree import ElementTree as ET

# === Hyperparameters for tuning ===
IOU_MATCH_THRESHOLD = 0.5
IOU_NMS_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.3
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# === Download Pascal VOC ===
def download_and_extract_voc2007(dest_dir='VOCdevkit'):
    voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, os.path.basename(voc_url))
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(voc_url, filepath)
    with tarfile.open(filepath) as tar:
        tar.extractall(path=dest_dir)

def find_voc_paths(base_dir="VOCdevkit"):
    for root, dirs, files in os.walk(base_dir):
        if "JPEGImages" in dirs and "Annotations" in dirs:
            return os.path.join(root, "JPEGImages"), os.path.join(root, "Annotations")
    raise FileNotFoundError("VOC paths not found")

def get_or_create_id_list(txt_path, jpeg_dir):
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return f.read().strip().split()
    ids = [f.split('.')[0] for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(ids))
    return ids

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in VOC_CLASSES:
            label = VOC_CLASSES.index(name)
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
    return boxes, labels

def generate_anchors(scales):
    return np.array([[x, y] for x in scales for y in scales])

def encode_boxes(gt_boxes, anchors):
    anchor_centers_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_centers_y = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    gt_centers_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_centers_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]

    tx = (gt_centers_x - anchor_centers_x) / anchor_widths
    ty = (gt_centers_y - anchor_centers_y) / anchor_heights
    tw = np.log(gt_widths / anchor_widths + 1e-7)
    th = np.log(gt_heights / anchor_heights + 1e-7)

    return np.stack([tx, ty, tw, th], axis=-1)

def decode_boxes(encoded_boxes, anchors):
    anchor_centers_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_centers_y = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    tx = encoded_boxes[:, 0]
    ty = encoded_boxes[:, 1]
    tw = encoded_boxes[:, 2]
    th = encoded_boxes[:, 3]

    pred_centers_x = tx * anchor_widths + anchor_centers_x
    pred_centers_y = ty * anchor_heights + anchor_centers_y
    pred_widths = np.exp(tw) * anchor_widths
    pred_heights = np.exp(th) * anchor_heights

    xmin = pred_centers_x - pred_widths / 2
    ymin = pred_centers_y - pred_heights / 2
    xmax = pred_centers_x + pred_widths / 2
    ymax = pred_centers_y + pred_heights / 2

    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

def compute_iou(boxes1, boxes2):
    boxes1 = np.expand_dims(boxes1, 1)
    boxes2 = np.expand_dims(boxes2, 0)

    x1 = np.maximum(boxes1[...,0], boxes2[...,0])
    y1 = np.maximum(boxes1[...,1], boxes2[...,1])
    x2 = np.minimum(boxes1[...,2], boxes2[...,2])
    y2 = np.minimum(boxes1[...,3], boxes2[...,3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[...,2] - boxes1[...,0]) * (boxes1[...,3] - boxes1[...,1])
    area2 = (boxes2[...,2] - boxes2[...,0]) * (boxes2[...,3] - boxes2[...,1])
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-7)

def match_anchors_to_gt(anchors, gt_boxes, gt_labels, iou_threshold=IOU_MATCH_THRESHOLD, num_classes=21):
    num_anchors = anchors.shape[0]
    matched_boxes = np.zeros((num_anchors, 4))
    matched_labels = np.zeros((num_anchors, num_classes))

    if len(gt_boxes) == 0:
        return matched_boxes, matched_labels

    ious = compute_iou(anchors, gt_boxes)
    max_iou_indices = np.argmax(ious, axis=1)
    max_ious = np.max(ious, axis=1)

    for anchor_idx in range(num_anchors):
        positive_gt_idxs = np.where(ious[anchor_idx] >= iou_threshold)[0]
        if len(positive_gt_idxs) == 0:
            continue
        best_gt_idx = max_iou_indices[anchor_idx]
        matched_boxes[anchor_idx] = gt_boxes[best_gt_idx]
        matched_labels[anchor_idx, gt_labels[best_gt_idx]] = 1

    matched_boxes = encode_boxes(matched_boxes, anchors)
    return matched_boxes, matched_labels

def load_data_with_anchor_matching(image_dir, annotation_dir, file_list, anchors, num_classes=21, input_size=(224,224)):
    images = []
    all_boxes = []
    all_labels = []

    anchors_norm = anchors / input_size[0]

    for file_id in file_list:
        img_path = os.path.join(image_dir, file_id + '.jpg')
        xml_path = os.path.join(annotation_dir, file_id + '.xml')
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, input_size)
        img = img / 255.0

        gt_boxes, gt_labels = parse_voc_annotation(xml_path)
        if len(gt_boxes) == 0:
            continue
        gt_boxes = np.array(gt_boxes) / input_size[0]

        matched_boxes, matched_labels = match_anchors_to_gt(anchors_norm, gt_boxes, np.array(gt_labels), IOU_MATCH_THRESHOLD, num_classes)

        images.append(img)
        all_boxes.append(matched_boxes)
        all_labels.append(matched_labels)

    return np.array(images), np.array(all_boxes), np.array(all_labels)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        bce = bce_fn(y_true, y_pred)  # shape: (batch, anchors, classes)
        
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        loss = alpha_factor * modulating_factor * bce
        
        # sum over classes, mean over batch and anchors
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        
        return loss
    return loss_fn



def ssd_model(input_shape=(224, 224, 3), num_anchors=27, num_classes=21):
    base = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    c3 = base.get_layer('conv3_block4_out').output
    c4 = base.get_layer('conv4_block6_out').output
    c5 = base.output
    p3 = layers.Conv2D(128, 1)(c3)
    p4 = layers.Conv2D(128, 1)(c4)
    p5 = layers.Conv2D(128, 1)(c5)
    p4 = layers.UpSampling2D()(p5) + p4
    p3 = layers.UpSampling2D()(p4) + p3
    x = layers.Concatenate()([
        layers.GlobalAveragePooling2D()(p3),
        layers.GlobalAveragePooling2D()(p4),
        layers.GlobalAveragePooling2D()(p5)])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    bbox = layers.Dense(num_anchors * 4)(x)
    bbox = layers.Reshape((num_anchors, 4), name="bbox")(bbox)
    cls = layers.Dense(num_anchors * num_classes, activation="sigmoid")(x)
    cls = layers.Reshape((num_anchors, num_classes), name="cls")(cls)
    return models.Model(inputs=base.input, outputs=[bbox, cls])

def nms_boxes(boxes, scores, iou_threshold=IOU_NMS_THRESHOLD, score_threshold=SCORE_THRESHOLD):
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=100,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)
    return selected_boxes.numpy(), selected_scores.numpy()

def visualize_detections(image, boxes, scores, labels, class_names, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()
    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = plt.Rectangle((x_min * image.shape[1], y_min * image.shape[0]),
                             width * image.shape[1], height * image.shape[0],
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_min * image.shape[1], y_min * image.shape[0] - 5,
                f"{class_names[label]}: {score:.2f}", color='yellow', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.show()

def giou_loss(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    y_true = tf.clip_by_value(y_true, 0, 1)
    y_pred = tf.clip_by_value(y_pred, 0, 1)

    x1 = tf.minimum(y_true[..., 0], y_true[..., 2])
    y1 = tf.minimum(y_true[..., 1], y_true[..., 3])
    x2 = tf.maximum(y_true[..., 0], y_true[..., 2])
    y2 = tf.maximum(y_true[..., 1], y_true[..., 3])

    px1 = tf.minimum(y_pred[..., 0], y_pred[..., 2])
    py1 = tf.minimum(y_pred[..., 1], y_pred[..., 3])
    px2 = tf.maximum(y_pred[..., 0], y_pred[..., 2])
    py2 = tf.maximum(y_pred[..., 1], y_pred[..., 3])

    inter_x1 = tf.maximum(x1, px1)
    inter_y1 = tf.maximum(y1, py1)
    inter_x2 = tf.minimum(x2, px2)
    inter_y2 = tf.minimum(y2, py2)

    inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)

    true_area = (x2 - x1) * (y2 - y1)
    pred_area = (px2 - px1) * (py2 - py1)

    union_area = true_area + pred_area - inter_area + 1e-7

    iou = inter_area / union_area

    enclose_x1 = tf.minimum(x1, px1)
    enclose_y1 = tf.minimum(y1, py1)
    enclose_x2 = tf.maximum(x2, px2)
    enclose_y2 = tf.maximum(y2, py2)

    enclose_area = tf.maximum(0.0, enclose_x2 - enclose_x1) * tf.maximum(0.0, enclose_y2 - enclose_y1) + 1e-7

    giou = iou - (enclose_area - union_area) / enclose_area

    loss = 1 - giou

    return tf.reduce_mean(loss)

if __name__ == '__main__':
    download_and_extract_voc2007()
    image_dir, annotation_dir = find_voc_paths()
    ids = get_or_create_id_list("VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", image_dir)
    anchors = generate_anchors([7, 4, 2])

    # Convert anchors to boxes (xmin,ymin,xmax,ymax) normalized [example scale 0.1]
    anchor_size = 0.1
    anchors_boxes = np.zeros((anchors.shape[0],4))
    anchors_boxes[:,0] = anchors[:,0]/224 - anchor_size/2
    anchors_boxes[:,1] = anchors[:,1]/224 - anchor_size/2
    anchors_boxes[:,2] = anchors[:,0]/224 + anchor_size/2
    anchors_boxes[:,3] = anchors[:,1]/224 + anchor_size/2

    images, boxes, labels = load_data_with_anchor_matching(image_dir, annotation_dir, ids, anchors_boxes)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat_labels = np.array([np.argmax(l) for l in labels])

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, strat_labels), 1):
        print(f"\n--- Fold {fold} ---")
        x_train, x_val = images[train_idx], images[val_idx]
        y_train_box, y_val_box = boxes[train_idx], boxes[val_idx]
        y_train_cls, y_val_cls = labels[train_idx], labels[val_idx]

        model = ssd_model(num_anchors=anchors_boxes.shape[0], num_classes=21)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={"bbox": giou_loss, "cls": focal_loss()},
            metrics={"cls": "accuracy"}
        )
        model.fit(
            x_train, {"bbox": y_train_box, "cls": y_train_cls},
            validation_data=(x_val, {"bbox": y_val_box, "cls": y_val_cls}),
            epochs=25, batch_size=8, verbose=2)

        preds_bbox, preds_cls = model.predict(x_val)

        flat_true = y_val_cls.reshape(-1, 21)
        flat_pred = preds_cls.reshape(-1, 21)
        ap = average_precision_score(flat_true, flat_pred, average='macro')
        print(f"Fold {fold} mAP: {ap:.4f}")

        for i in range(3):
            p, r, _ = precision_recall_curve(flat_true[i], flat_pred[i])
            plt.plot(r, p, label=f"Sample {i}")
        plt.title(f"PR Curve Fold {fold}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

        flipped = tf.image.flip_left_right(x_val)
        tta_bbox, tta_cls = model.predict(flipped)
        tta_acc = np.mean((tta_cls > SCORE_THRESHOLD).astype(int) == y_val_cls)
        print(f"Fold {fold} TTA Accuracy: {tta_acc:.4f}")

        for i in np.random.choice(len(x_val), 5, replace=False):
            img = x_val[i]
            boxes_i = preds_bbox[i]
            class_probs = preds_cls[i]

            max_scores = class_probs.max(axis=-1)
            max_labels = class_probs.argmax(axis=-1)

            mask = max_scores > SCORE_THRESHOLD
            filtered_boxes = boxes_i[mask]
            filtered_scores = max_scores[mask]
            filtered_labels = max_labels[mask]

            selected_boxes, selected_scores = nms_boxes(filtered_boxes, filtered_scores,
                                                        iou_threshold=IOU_NMS_THRESHOLD,
                                                        score_threshold=SCORE_THRESHOLD)
            selected_labels = filtered_labels[:len(selected_boxes)]

            visualize_detections(img, selected_boxes, selected_scores, selected_labels, VOC_CLASSES)
