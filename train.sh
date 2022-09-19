#多GPU训练
# 使用coco进行预训练
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 8 --device 2,3 --sync-bn --batch_size 128 --data data/coco.yaml --img 320 320 --cfg cfg/training/yolov7-tiny.yaml --weights "" --name yolov7-test --hyp data/hyp.scratch.tiny.yaml
# 在coco基础上进行微调
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 8 --device 0,3 --sync-bn --batch_size 128 --data data/fms.yaml --img 320 320 --cfg cfg/training/yolov7-tiny.yaml --weights runs/train/yolov7-coco32/weights/best.pt --name yolov7-fms-fresh --hyp data/hyp.scratch.tiny.yaml --single_cls
# 融合训练
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9528 train.py --workers 8 --device 2,3 --sync-bn --batch_size 128 --data data/fms.yaml --img 320 320 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name yolov7-fms-mix --hyp data/hyp.scratch.tiny.yaml # --resume 
# review测试
#python -m torch.distributed.launch --nproc_per_node 4 --master_port 9528 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch_size 128 --data data/fms.yaml --img 320 320 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name yolov7-fms-test-review --hyp data/hyp.scratch.tiny.yaml # --resume 
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 9528 train.py --workers 8 --device 2,3 --sync-bn --batch_size 128 --data data/fms.yaml --img 320 320 --cfg cfg/training/yolov7-tiny-fms.yaml --name yolov7-fms-laydown --hyp data/hyp.scratch.tiny.yaml --noautoanchor --epochs 100  --weights 'runs/train/yolov7-fms-test-review64/weights/best.pt' # --resume
# fms.yaml 修改数据路径
# 确定anchors的继承自与训练 --resume,--noautoanchor, 不行就修改yolov7-tiny.yaml



# 单GPU训练
#python train.py --workers 8 --device 0 --batch-size 32 --data data/fms.yaml --img 320 240 --cfg cfg/training/yolov7-tiny.yaml --weights runs/train/yolov7-coco32/weights/best.pt --name yolov7-fms-singlecls --hyp data/hyp.scratch.tiny.yaml
