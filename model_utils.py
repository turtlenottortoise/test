import os, tempfile, cv2, torch, pandas as pd
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.visualization import PoseLocalVisualizer            # facultatif

# ───────────── chemins relatifs ─────────────
ROOT = os.path.dirname(__file__)
DET_CFG  = os.path.join(ROOT, "config", "rtmdet_s.py")
POSE_CFG = os.path.join(ROOT, "config", "rtmpose_m.py")          # non obligatoire, mais utile
DET_PTH  = os.path.join(ROOT, "checkpoints", "rtmdet_s.pth")
POSE_PTH = os.path.join(ROOT, "checkpoints", "rtmpose_m.pth")

# ───────────── lazy‑load global ─────────────
_DET = _POSE_INFER = _VIS = None
def _lazy_load(device="cuda" if torch.cuda.is_available() else "cpu"):
    global _DET, _POSE_INFER, _VIS
    if _DET and _POSE_INFER:
        return _DET, _POSE_INFER, _VIS

    init_default_scope("mmpose")
    _DET = init_detector(DET_CFG, DET_PTH or None, device=device)

    # MMPoseInferencer accepte soit un alias (ex. "human") soit un vrai config+ckpt
    _POSE_INFER = MMPoseInferencer(
        pose2d='human',                 # alias simple
        pose_weight=POSE_PTH if os.path.exists(POSE_PTH) else None,
        device=device
    )

    # Visualisateur (facultatif)
    _VIS = PoseLocalVisualizer()        # couleurs, squelette, etc.
    _VIS.set_dataset_meta(_POSE_INFER.dataset_meta)

    return _DET, _POSE_INFER, _VIS

# ───────────── pipeline principal ─────────────
def process_video(path: str):
    det, pose_infer, vis = _lazy_load()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.mkdtemp()
    out_mp4 = os.path.join(tmp, "annotated.mp4")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rows, frame_idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Détection personnes → bbox
        det_res = inference_detector(det, frame)
        bboxes = det_res.pred_instances.bboxes[
            det_res.pred_instances.labels == 0
        ]

        # 2) Pose : nouvelle API
        preds = next(
            pose_infer(frame, det_result=bboxes, bbox_format="xyxy")
        )["predictions"]  # list[dict]

        # 3) Visualisation (optionnel)
        vis_frame = vis.draw_pose(frame.copy(), preds)["img"]
        writer.write(vis_frame)

        # 4) Stockage CSV
        for pid, person in enumerate(preds):
            for k, (x, y, s) in enumerate(person["keypoints"]):
                rows.append({
                    "frame": frame_idx,
                    "person": pid,
                    "kp": k,
                    "x": float(x), "y": float(y), "score": float(s)
                })
        frame_idx += 1

    cap.release(); writer.release()
    csv_path = os.path.join(tmp, "keypoints.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return out_mp4, csv_path
