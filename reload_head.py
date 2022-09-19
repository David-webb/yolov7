import torch
import torch.nn as nn
from models.experimental import attempt_load, reload_head

# class Ensemble(nn.ModuleList):
    # # Ensemble of models
    # def __init__(self):
        # super(Ensemble, self).__init__()

    # def forward(self, x, augment=False):
        # y = []
        # for module in self:
            # y.append(module(x, augment)[0])
        # # y = torch.stack(y).max(0)[0]  # max ensemble
        # # y = torch.stack(y).mean(0)  # mean ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        # return y, None  # inference, train output

# def attempt_load(weights, map_location=None):
    # # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    # model = Ensemble()
    # for w in weights if isinstance(weights, list) else [weights]:
        # # attempt_download(w)
        # print(w)
        # ckpt = torch.load(w, map_location=map_location)  # load
        # model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # # Compatibility updates
    # for m in model.modules():
        # if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            # m.inplace = True  # pytorch 1.7.0 compatibility
        # elif type(m) is nn.Upsample:
            # m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        # elif type(m) is Conv:
            # m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    # print(model.modules())
    # if len(model) == 1:
        # return model[-1]  # return model
    # else:
        # print('Ensemble created with %s\n' % weights)
        # for k in ['names', 'stride']:
            # setattr(model, k, getattr(model[-1], k))
        # return model  # return ensemble

def rebuild_model_shell():
    """重新构建一个类别个数为1的Model对象, 将训练好的模型参数加载进去
    但是头部的三个卷积参数要裁剪一下
        torch.Size([1, 255, 1, 1]) # (1,3*85,1,1)
        torch.Size([1, 255, 1, 1])
        torch.Size([1, 255, 1, 1])
        85: 5(xywh, score) + 80(cls score), 3: 每个grid设置的anchors的个数
    ====>
        torch.Size([1, 18, 1, 1])
        torch.Size([1, 18, 1, 1])
        torch.Size([1, 18, 1, 1])
        18: 5(xywh, score) + 1(cls score), 3: 每个grid设置的anchors的个数

    =====>
    77.m.0.weight torch.Size([255, 128, 1, 1])
    77.m.0.bias torch.Size([255])
    77.m.1.weight torch.Size([255, 256, 1, 1])
    77.m.1.bias torch.Size([255])
    77.m.2.weight torch.Size([255, 512, 1, 1])
    77.m.2.bias torch.Size([255])
    77.ia.0.implicit torch.Size([1, 128, 1, 1])
    77.ia.1.implicit torch.Size([1, 256, 1, 1])
    77.ia.2.implicit torch.Size([1, 512, 1, 1])
    77.im.0.implicit torch.Size([1, 255, 1, 1])
    77.im.1.implicit torch.Size([1, 255, 1, 1])
    77.im.2.implicit torch.Size([1, 255, 1, 1])

    """
    pass

if __name__ == "__main__":
    # model_fp = "./runs/train/yolov7-fms-laydown16/weights/best.pt"
    model_fp = "./weights/mix-01.pt"
    device = torch.device('cuda:2')
    # attempt_load(model_fp, device)
    reload_head(model_fp, device)
    pass
