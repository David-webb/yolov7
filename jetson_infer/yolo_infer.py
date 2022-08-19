"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-08-17 15:03
 * Filename      : yolo_infer.py
 * Description   : 
"""
import os
import time
import tensorrt as trt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2

# from .trt_infer_with_torchstream import allocate_buffers, do_inference

class YoloDet():
    def __init__(self, trt_path, imgsz, device):
        """
        Args:
            imgsz: int，输入图像默认为方形，例如，320表示(320,320)
        """
        self.conf = 0.3
        self.n_classes = 1
        self.class_names = ['person']
        self.imgsz = imgsz
        self.device = device
        # detect中pad的做法，320,240的输入默认是被pad成320,320？ # 答：pad成320,256的,但后期会更改
        # self.pad_board = torch.ones(1, self.imgsz, self.imgsz,3, dtype=torch.float16) * 114.0
        # self.pad_board[:] = 114.0 # 灰色处理
        self.init_trt_infer_env(trt_path)
        pass

    def torch_device_from_trt(self, device):
        if device == trt.TensorLocation.DEVICE:
            return torch.device("cuda")
        elif device == trt.TensorLocation.HOST:
            return torch.device("cpu")
        else:
            return TypeError("%s is not supported by torch" % device)


    def torch_dtype_from_trt(self, dtype):
        if dtype == trt.int8:
            return torch.int8
        elif dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError("%s is not supported by torch" % dtype)

    def allocate_buffers(self, engine):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        # stream = cuda.Stream()
        self.stream = torch.cuda.current_stream().cuda_stream
        # print('engine.max_batch_size', engine.max_batch_size)
        # ============ 官方代码 ================
        # for binding in engine:
            # size = trt.volume(engine.get_binding_shape(binding))
            # dtype = trt.nptype(engine.get_binding_dtype(binding))
            # host_mem = cuda.pagelocked_empty(size, dtype)
            # device_mem = cuda.mem_alloc(host_mem.nbytes)
            # self.bindings.append(int(device_mem))
            # if engine.binding_is_input(binding):
                # self.inputs.append({'host': host_mem, 'device': device_mem})
            # else:
                # self.outputs.append({'host': host_mem, 'device': device_mem})
        # ======================================
        for binding in engine:
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.bindings.append(None)
                idx = engine.get_binding_index(binding)
                self.inputs.append(idx)
            else:
                size = (engine.max_batch_size,) + \
                    tuple(engine.get_binding_shape(binding))
                dtype = self.torch_dtype_from_trt(engine.get_binding_dtype(binding))
                device = self.torch_device_from_trt(engine.get_location(binding))
                # print("output device", device)
                device_mem_torch = torch.empty(size=size, dtype=dtype, device=device)
                self.bindings.append(device_mem_torch.data_ptr())
                self.outputs.append(device_mem_torch)
                # outputs.append(HostDeviceMem(host_mem, device_mem))
        # return inputs, outputs, bindings, stream

    def init_trt_infer_env(self, trt_path):
        """初始化TensotRT的推理环境
        """
        print("创建Logger")
        TRT_LOGGER = trt.Logger()
        # time.sleep(20)
        print("创建Runtime")
        runtime = trt.Runtime(TRT_LOGGER)
        # time.sleep(20)
        print("读取并反序列换engine")
        with open(trt_path, 'rb')as rd:
            self.trt_engine = runtime.deserialize_cuda_engine(rd.read())
        # time.sleep(20)
        print("创建context")
        self.context = self.trt_engine.create_execution_context()
        # time.sleep(20)
        print("allocate buffer")
        # self.blazes = trt_utils.allocate_buffers(self.trt_engine)
        self.blazes = self.allocate_buffers(self.trt_engine)
        # time.sleep(20)
        print("finish allocating buffer")

    def do_inference(self, image, batch_size=1):
        # Transfer input data to the GPU.
        # print(len(inputs))
        # Run inference.
        # 用输入的image内存指针更新bindings
        # 这里为什么不绑定固定的输入冲区?因为，数据的复制比指针的赋值要来的慢
        # ========= 修正：由于单次推理只有一个输入，顾不需要为image设置list(实际上,gpu tensor放入list中后，tensor仍然在gpu上，猜测应该是地址放入了list. 但是如果通过list访问该tensor，需要切换到cpu环境，取得地址后，进一步访问，这里会涉及上下文的切换，会增加开销。
        # for i, idx in enumerate(inputs):
            # bindings[idx] = images[i].contiguous().data_ptr()
        # ========= 修正后，单个image直接赋值
        self.bindings[self.inputs[0]] = image.data_ptr() # 只适用于单个输入的模型

        # 执行推理
        self.context.execute_async(batch_size=batch_size,
                              bindings=self.bindings, stream_handle=self.stream)
        
        # 从GPU显存提取推理后的结果
        outputs = tuple(self.outputs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs
        

    def torch_device_from_trt(self, device):
        if device == trt.TensorLocation.DEVICE:
            return torch.device("cuda")
        elif device == trt.TensorLocation.HOST:
            return torch.device("cpu")
        else:
            return TypeError("%s is not supported by torch" % device)


    # def run_trt_inference(self, image):
        # # inputs_b, outputs_b, bindings_b, stream_b = self.blazes
        # # images = [image]
        # reg_out = self.do_inference(image)

        # # print(reg_out.size())
        # return None, reg_out_new, None

    def run(self, frame, conf=0.5, end2end=False):
        """
        Args:
            frame: numpy格式的视频帧或图片, 按照opencv的读取方式，shape = (h,w,c)
        Returns:
            final_rets, 字典类型， {'nobody':True} 或者 {'bbox': [xyxy]}
        """
        # assert isinstance(frame. np.ndarray), "frame type must be numpy array(from opencv)" # 实际推理时，默认该类型，可省略
        frame_ready = self.preprocess(frame)
        det_out = self.do_inference(frame_ready)
        # print(det_out, type(det_out), det_out.size())

        data = det_out.cpu().numpy()

        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            # print(self.n_classes)
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio=1.0)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
            origin_img = vis(frame, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
            self.save_img(origin_img)
        return origin_img

        pass
    
    def preprocess(self, frame):
        """#前处理
        步骤：
            RGB -> BGR
            转到torch Gpu-tensor
            将frame pad到指定的尺寸, inp_frame
            对inp_frame 归一化
            inp_frame的转置(2,0,1)
            内存连续
        """
        # img 从numpy转到torch
        # pad
        inp_height, inp_width = frame.shape[:2]
        inp_image = torch.from_numpy(frame[:,:,::-1].copy()).to(self.device) # RGB->BGR && numpy->torch
        inp_image = F.pad(inp_image, (0,0,0,0,40,40), value=114.0)
        # inp_image = ((inp_image / 255. - self.mean_torch) / self.std_torch).type(torch.float32)
        inp_image = inp_image / 255.0 # mean和std为None
        # print(inp_image.size())
        # 转成CHW的形式
        image = inp_image.permute(2, 0, 1).view(1,3,self.imgsz,self.imgsz).contiguous()
        # pad_board[:,]
        return image

        pass


    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)


    def postprocess(self, predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2 - 40.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2 - 40.
        boxes_xyxy /= ratio
        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

        pass

    def save_img(self, img, img_fp=None):
        if img_fp:
            prefix, tail = img_fp.split(".")
            save_fp = prefix + "_ans."+tail
        else:
            save_fp = "./default.jpg"
        cv2.imwrite(save_fp, img)
        pass
    



_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def try_yolo_trt():
    #读取图片或者视频帧
    pass

if __name__ == "__main__":
    device = 'cuda'
    trt_path = './yolov7-tiny-fms.trt'
    imgsz = 320
    ydh = YoloDet(trt_path, imgsz, device)
    
    # 读取图片，执行推理
    img = cv2.imread('./10086.jpg')
    ydh.run(img)
