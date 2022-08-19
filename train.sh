#多GPU训练
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 8 --device 2,3 --sync-bn --batch_size 128 --data data/fms.yaml --img 320 320 --cfg cfg/training/yolov7-tiny.yaml --weights runs/train/yolov7-coco32/weights/best --name yolov7-fms-singlecls --hyp data/hyp.scratch.tiny.yaml --single_cls
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 8 --device 2,3 --sync-bn --batch_size 128 --data data/fms.yaml --img 320 240 --cfg cfg/training/yolov7-tiny.yaml --weights runs/train/yolov7-coco32/weights/best.pt --name yolov7-fms-test --hyp data/hyp.scratch.tiny.yaml --single_cls

# 单GPU训练
#python train.py --workers 8 --device 0 --batch-size 32 --data data/fms.yaml --img 320 240 --cfg cfg/training/yolov7-tiny.yaml --weights runs/train/yolov7-coco32/weights/best.pt --name yolov7-fms-singlecls --hyp data/hyp.scratch.tiny.yaml
