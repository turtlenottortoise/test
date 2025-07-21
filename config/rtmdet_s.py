_base_ = [
    "mmdet::_base_/models/rtmdet_s.py",
    "mmdet::_base_/datasets/coco_detection.py",
    "mmdet::_base_/default_runtime.py",
]
model = dict(bbox_head=dict(num_classes=1))
train_dataloader = val_dataloader = test_dataloader = None
