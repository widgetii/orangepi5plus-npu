"""
Post-processing functions for RKNN model outputs.
Pure numpy — no torch dependency.
"""
import numpy as np

# ── Classification ──────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def classify_top_k(output, k=5):
    """Return top-k class indices.
    Some RKNN models (e.g. MBv1) export with softmax already applied, others
    (e.g. ResNet50) export logits. Either way, argmax is the same — we just
    use the raw values directly. If the values look like logits (max > 1),
    we still return them as 'scores' for display."""
    flat = output.flatten()
    top_k = np.argsort(flat)[-k:][::-1]
    return {
        "classes": top_k.tolist(),
        "scores": flat[top_k].tolist(),
        "top1_class": int(top_k[0]),
        "top1_score": float(flat[top_k[0]]),
    }

# ── Detection helpers ───────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep)

def per_class_nms(boxes, classes, scores, iou_thresh):
    keep_boxes, keep_cls, keep_scores = [], [], []
    for c in np.unique(classes):
        mask = classes == c
        b, s = boxes[mask], scores[mask]
        k = nms(b, s, iou_thresh)
        if len(k):
            keep_boxes.append(b[k])
            keep_cls.append(np.full(len(k), c, dtype=int))
            keep_scores.append(s[k])
    if not keep_boxes:
        return np.zeros((0, 4)), np.array([], dtype=int), np.array([])
    return np.vstack(keep_boxes), np.concatenate(keep_cls), np.concatenate(keep_scores)

# ── YOLOv5 ──────────────────────────────────────────────────────────────

YOLOV5_ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],      # stride 8  (80×80)
    [[30, 61], [62, 45], [59, 119]],      # stride 16 (40×40)
    [[116, 90], [156, 198], [373, 326]],  # stride 32 (20×20)
]

def yolov5_postprocess(outputs, img_size=(640, 640),
                       conf_thresh=0.25, nms_thresh=0.45,
                       anchors=None):
    """
    outputs: list of 3 float32 arrays from RKNN.
    Each output dims = [1, H, W, 255] (NHWC, last dim = 3*(5+80)=255).
    The RKNN export bakes sigmoid into the model, so values are already in [0,1].
    Strides are inferred from spatial size: 80=>8, 40=>16, 20=>32.
    """
    if anchors is None:
        anchors = YOLOV5_ANCHORS

    all_boxes, all_classes, all_scores = [], [], []

    for out in outputs:
        out = out.squeeze(0) if out.ndim == 4 else out  # [H, W, 255]
        h, w, c = out.shape
        num_anchors = 3
        nc = c // num_anchors - 5  # 80
        stride = img_size[0] // h
        # Pick the right anchor set based on stride
        if stride == 8:
            anc_set = anchors[0]
        elif stride == 16:
            anc_set = anchors[1]
        else:
            anc_set = anchors[2]

        # Reshape [H, W, 255] -> [H, W, 3, 85]
        out = out.reshape(h, w, num_anchors, 5 + nc)

        # Grid
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid = np.stack([gx, gy], axis=-1).astype(np.float32)  # [H, W, 2]

        for a_idx in range(num_anchors):
            slab = out[:, :, a_idx, :]  # [H, W, 85]
            xy = slab[..., :2]
            wh = slab[..., 2:4]
            obj = slab[..., 4]
            cls = slab[..., 5:]

            anchor_w, anchor_h = anc_set[a_idx]

            # Decode (no sigmoid — already applied)
            bxy = (xy * 2 - 0.5 + grid) * stride
            bwh = ((wh * 2) ** 2) * np.array([anchor_w, anchor_h], dtype=np.float32)

            x1y1 = bxy - bwh / 2
            x2y2 = bxy + bwh / 2
            boxes = np.concatenate([x1y1, x2y2], axis=-1).reshape(-1, 4)

            obj_flat = obj.reshape(-1)
            cls_flat = cls.reshape(-1, nc)

            class_max = np.max(cls_flat, axis=1)
            combined = obj_flat * class_max
            class_ids = np.argmax(cls_flat, axis=1)

            mask = combined >= conf_thresh
            all_boxes.append(boxes[mask])
            all_classes.append(class_ids[mask])
            all_scores.append(combined[mask])

    if not all_boxes or all(len(b) == 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([], dtype=int), np.array([])

    boxes = np.vstack(all_boxes)
    classes = np.concatenate(all_classes)
    scores = np.concatenate(all_scores)
    return per_class_nms(boxes, classes, scores, nms_thresh)

# ── YOLOv8 ──────────────────────────────────────────────────────────────

def _dfl(x, reg_max=16):
    """Distribution Focal Loss decode: softmax over bins, weighted sum."""
    n = x.shape[0]
    x = x.reshape(n, 4, reg_max)
    e = np.exp(x - np.max(x, axis=2, keepdims=True))
    w = e / e.sum(axis=2, keepdims=True)
    arange = np.arange(reg_max, dtype=np.float32)
    return (w * arange).sum(axis=2)

def yolov8_postprocess(outputs, img_size=(640, 640),
                       conf_thresh=0.25, nms_thresh=0.45):
    """
    outputs: list of 9 float32 arrays from RKNN YOLOv8 export.
    Triplet order per scale: (box [1,64,H,W], cls [1,80,H,W], score_sum [1,1,H,W]).
    Scales by spatial size: 80×80 (stride 8), 40×40 (stride 16), 20×20 (stride 32).
    The score_sum tensor (3rd of each triplet) is unused — kept for ONNX export
    compatibility. Class scores are already sigmoid-applied by the export.
    """
    strides_by_h = {80: 8, 40: 16, 20: 32}
    all_boxes, all_classes, all_scores = [], [], []

    n_per = 3  # box, cls, score_sum
    n_branches = len(outputs) // n_per

    for i in range(n_branches):
        box_out = outputs[i * n_per]      # [1, 64, H, W]
        cls_out = outputs[i * n_per + 1]  # [1, 80, H, W]

        # NCHW from rknn_query — the reshape using query'd dims puts data
        # in [1, C, H, W] order. Squeeze batch.
        if box_out.ndim == 4:
            box_out = box_out[0]  # [64, H, W]
        if cls_out.ndim == 4:
            cls_out = cls_out[0]  # [80, H, W]

        c_box, h, w = box_out.shape
        nc = cls_out.shape[0]
        stride = strides_by_h.get(h, img_size[0] // h)
        reg_max = c_box // 4  # 16

        # Grid (HW)
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.float32)

        # DFL decode boxes: [64, H, W] -> [H*W, 64] -> [H*W, 4]
        box_flat = box_out.reshape(c_box, -1).T  # [H*W, 64]
        dist = _dfl(box_flat, reg_max)            # [H*W, 4]

        x1y1 = (grid + 0.5 - dist[:, :2]) * stride
        x2y2 = (grid + 0.5 + dist[:, 2:]) * stride
        boxes = np.concatenate([x1y1, x2y2], axis=1)  # [H*W, 4]

        # Classes (already sigmoid-applied by export)
        cls_flat = cls_out.reshape(nc, -1).T  # [H*W, 80]
        class_ids = np.argmax(cls_flat, axis=1)
        max_scores = np.max(cls_flat, axis=1)

        mask = max_scores >= conf_thresh
        all_boxes.append(boxes[mask])
        all_classes.append(class_ids[mask])
        all_scores.append(max_scores[mask])

    if not all_boxes or all(len(b) == 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([], dtype=int), np.array([])

    boxes = np.vstack(all_boxes)
    classes = np.concatenate(all_classes)
    scores = np.concatenate(all_scores)
    return per_class_nms(boxes, classes, scores, nms_thresh)

# ── DeepLabv3 ───────────────────────────────────────────────────────────

def deeplabv3_postprocess(output, target_size=(513, 513)):
    """
    output: float32 [1, H, W, C]. The RKNN export reports dims as
    [1, 65, 65, 21] in NHWC and the data is laid out HWC after squeeze.
    Returns: int32 [target_H, target_W] class map.
    """
    out = output.squeeze(0)  # [H, W, C]
    h, w, c = out.shape

    # If channel count is much larger than H, that's CHW — flip to HWC
    if c > h * 2:
        out = out.transpose(1, 2, 0)
        h, w, c = out.shape

    th, tw = target_size

    if h != th or w != tw:
        row_idx = (np.arange(th) * h / th).astype(int).clip(0, h - 1)
        col_idx = (np.arange(tw) * w / tw).astype(int).clip(0, w - 1)
        out = out[np.ix_(row_idx, col_idx)]

    return np.argmax(out, axis=-1).astype(np.int32)

# ── COCO class names ────────────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
