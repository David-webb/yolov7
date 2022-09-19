import os
import numpy as np
import random
import torch
import torch.nn as nn
from models.experimental import *
from models.common import Conv, DWConv, ImplicitM
from utils.google_utils import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output





class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                max_output_boxes_per_class=torch.tensor([100]),
                iou_threshold=torch.tensor([0.45]),
                score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)


class TRT_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh # if max_wh != 0 : non-agnostic else : agnostic
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        scores *= conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold)
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.cat([X, selected_boxes, selected_categories, selected_scores], 1)

class ONNX_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        scores *= conf
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(boxes, scores, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        self.model = model.to(device)
        self.model.model[-1].end2end = True
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x



def attempt_load_fms(weights, map_location=None):
    """针对fms项目的模型裁剪：主要是head裁剪
    """
    pass

def reload_head(weights, map_location=None):
    """
    参数路径：
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

    结构路径：
	取ckpt[77]: IDetect(
          (m): ModuleList(
            (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
          )
          (ia): ModuleList(
            (0): ImplicitA()
            (1): ImplicitA()
            (2): ImplicitA()
          )
          (im): ModuleList(
            (0): ImplicitM()
            (1): ImplicitM()
            (2): ImplicitM()
          )
        )
    """
    # 构建保存路径
    work_dir, pt_name  = os.path.split(weights) 
    pt_name = pt_name.split(".")[0] + "_sim.pt"
    pt_save = os.path.join(work_dir, pt_name)

    w = weights
    ckpt = torch.load(w, map_location=map_location)  # load

    # ========================= 模板 =========================== 
    # 加载模型参数
    pmodel = ckpt['ema'].model.named_parameters()
    # pmodel = ckpt['model'].named_parameters()
    
    sim_head = {}
    for name, p in pmodel:
        # print(name, p.size())
        if '77' in name:
            sim_head[name] = p

    # print(pmodel.77)

    # # 加载模型结构
    # tmodel = ckpt['ema'].model
    # # print("取ckpt[76]:",tmodel[77].m[0])
    # print("取ckpt[77]:",tmodel[77].m)
    # tmodel[77].m[0] = nn.Conv2d(128, 18, 1)
    # print("取ckpt[76]:",tmodel[77].m[0].weight.size())

    # tmodel[77].m[0].weight = torch.nn.Parameter(torch.ones((18,128,1,1)), requires_grad=False)
    # print("取ckpt[76]:",tmodel[77].m[0].weight.size())
    # ===========================================================
    for i in range(3):
        print((ckpt['ema'].model[77].m[i].weight).requires_grad)
        print((ckpt['ema'].model[77].im[i].implicit).requires_grad)
    # 修改head
    ckpt['ema'].model[77].m[0] = nn.Conv2d(128,18,1)
    ckpt['ema'].model[77].m[1] = nn.Conv2d(256,18,1)
    ckpt['ema'].model[77].m[2] = nn.Conv2d(512,18,1)
    ckpt['ema'].model[77].im[0] = ImplicitM(18) 
    ckpt['ema'].model[77].im[1] = ImplicitM(18)
    ckpt['ema'].model[77].im[2] = ImplicitM(18)

    # 修改参数
    # print(type(sim_head['77.m.0.weight'].view((3,85, 128, 1, 1))[:,:6,:,:,:]))
    # print((sim_head['77.m.0.weight'].view((3,85, 128, 1, 1))[:,:6,:,:,:]).contiguous().view(18,128,1,1).size())
    tmp_ = sim_head['77.m.0.weight'].view((3,85, 128, 1, 1))[:,:6,:,:,:].contiguous().view(18,128,1,1)
    ckpt['ema'].model[77].m[0].weight = torch.nn.Parameter(tmp_, requires_grad=False)
    tmp_ = sim_head['77.m.0.bias'].view((3,85))[:,:6].contiguous().view(18)
    ckpt['ema'].model[77].m[0].bias = torch.nn.Parameter(tmp_, requires_grad=False)

    tmp_ = sim_head['77.m.1.weight'].view((3,85, 256, 1, 1))[:,:6,:,:,:].contiguous().view(18,256,1,1)
    ckpt['ema'].model[77].m[1].weight = torch.nn.Parameter(tmp_, requires_grad=False) 
    tmp_ = sim_head['77.m.1.bias'].view((3,85))[:,:6].contiguous().view(18)
    ckpt['ema'].model[77].m[1].bias = torch.nn.Parameter(tmp_, requires_grad=False)

    tmp_ = sim_head['77.m.2.weight'].view((3,85,512, 1, 1))[:,:6,:,:,:].contiguous().view(18,512,1,1)
    ckpt['ema'].model[77].m[2].weight = torch.nn.Parameter(tmp_, requires_grad=False)
    tmp_ = sim_head['77.m.2.bias'].view((3,85))[:,:6].contiguous().view(18)
    ckpt['ema'].model[77].m[2].bias = torch.nn.Parameter(tmp_, requires_grad=False) 

    tmp_ = sim_head['77.im.0.implicit'].view((3,85))[:, :6].contiguous().view(18)
    ckpt['ema'].model[77].im[0].implicit.weight = torch.nn.Parameter(tmp_, requires_grad=False)
    tmp_ = sim_head['77.im.1.implicit'].view((3,85))[:, :6].contiguous().view(18)
    ckpt['ema'].model[77].im[1].implicit.weight = torch.nn.Parameter(tmp_, requires_grad=False) 
    tmp_ = sim_head['77.im.2.implicit'].view((3,85))[:, :6].contiguous().view(18)
    ckpt['ema'].model[77].im[2].implicit.weight = torch.nn.Parameter(tmp_, requires_grad=False)
    
    ckpt['ema'].model = ckpt['ema'].model.half()
    
    # 成员变量参数修改
    ckpt['ema'].yaml['nc'] = 1
    ckpt['ema'].model[77].nc = 1
    ckpt['ema'].model[77].no = 6
    
    torch.save(ckpt, pt_save)
    pass


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        print("ckpt的keys:", ckpt.keys()) # ckpt的keys: dict_keys(['epoch', 'best_fitness', 'training_results', 'model', 'ema', 'updates', 'optimizer', 'wandb_id'])
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # # print(ckpt['model'])
    # pmodel = dict(ckpt['ema'].model.named_parameters())
    # for name, p in pmodel.items():
        # print(name, p.size())
    # print(pmodel["77.ia.0.implicit"].size())


    # tmodel = ckpt['ema'].model
    # # print("取ckpt[76]:",tmodel[77].m[0])
    # print("取ckpt[77]:",tmodel[77])
    # tmodel[77].m[0] = nn.Conv2d(128, 18, 1)
    # print("取ckpt[76]:",tmodel[77])
    # # print("取ckpt[76]:",tmodel[77].m[0].weight)
    # # tmodel[77].m[0].weight = tmodel[77].m[0].weight

    # Compatibility updates
    for m in model.modules():
        # print(m)
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        # print(model[-1])
        # print(model[-1]["IDetect"])
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


