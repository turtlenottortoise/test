import os, tempfile, cv2, torch, pandas as pd
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result

DIR = os.path.dirname(__file__)
DET_CFG  = os.path.join(DIR, "config", "rtmdet_s.py")
POSE_CFG = os.path.join(DIR, "config", "rtmpose_m.py")
DET_PTH  = os.path.join(DIR, "checkpoints", "rtmdet_s.pth")
POSE_PTH = os.path.join(DIR, "checkpoints", "rtmpose_m.pth")

_DET = _POSE = None
def _load(device="cuda" if torch.cuda.is_available() else "cpu"):
    global _DET, _POSE
    if _DET: return _DET, _POSE
    init_default_scope("mmpose")
    _DET  = init_detector(DET_CFG,  DET_PTH  if os.path.exists(DET_PTH)  else None, device)
    _POSE = init_pose_model(POSE_CFG, POSE_PTH if os.path.exists(POSE_PTH) else None, device)
    return _DET, _POSE

def process_video(path):
    det, pose = _load()
    cap = cv2.VideoCapture(path)
    fps, w, h = (cap.get(cv2.CAP_PROP_FPS) or 25,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    tmp = tempfile.mkdtemp()
    out_vid = os.path.join(tmp, "annotated.mp4")
    writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows, f = [], 0
    while True:
        ok, frame = cap.read();  f += 1
        if not ok: break
        det_res = inference_detector(det, frame)
        bboxes = det_res.pred_instances.bboxes[det_res.pred_instances.labels == 0]
        poses  = inference_top_down_pose_model(pose, frame,
                 [{"bbox": b} for b in bboxes], bbox_format="xyxy", dataset="coco")
        visf = vis_pose_result(pose, frame, poses, radius=3, thickness=2, show=False)
        writer.write(visf)
        for pid, res in enumerate(poses):
            for k, (x, y, s) in enumerate(res["keypoints"]):
                rows.append({"frame": f, "person": pid, "kp": k, "x": x, "y": y, "score": s})
    cap.release(); writer.release()
    csv_path = os.path.join(tmp, "keypoints.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return out_vid, csv_path
