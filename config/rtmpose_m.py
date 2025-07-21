_base_ = [
    "mmpose::_base_/default_runtime.py",
    "mmpose::_base_/datasets/coco_wholebody.py",
    "mmpose::_base_/models/rtmpose/rtmpose_m_8xb256-420e_coco-wholebody-256x192.py",
]
model = dict(test_cfg=dict(output_heatmaps=True))
