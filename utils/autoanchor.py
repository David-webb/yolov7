# Auto-anchor utils

import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm

from utils.general import colorstr


def check_anchor_order(m):
    """
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    检查anchor的顺序是否和head_layer匹配，不匹配就进行修正
    """
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area # (9,) 所有anchor的面积
    da = a[-1] - a[0]  # delta a # anchors面积最大head_layer 减去面积最小的anchors
    ds = m.stride[-1] - m.stride[0]  # deltas # model（head_layers）中最大的stride减去最小的stride 
    if da.sign() != ds.sign():  # same order # sign是符号函数
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """# Check anchor fit to data, recompute if necessary
    检查预设的anchors和数据集中的target的匹配程度，不满足条件就kmeans聚类重构anchors
    具体步骤如下：
        先将所有targets的尺寸放缩到inp尺寸上（不是resize，是保持横纵比的缩放）
        然后使用metric函数计算所有target与当前anchors的匹配度(bpr和aat指标)
        如果bpr指标没有达到0.98,就是用k-means对所有targets进行聚类生成一套新的anchors
        再用metric对新生成的anchors计算匹配度
            如果新的bpr指标有提升，就用新的anchors,(check_anchor_order验证新的anchors是否和head_layer匹配，有误纠正)
            否则，就延续使用原来预设的anchors

    关于metric指标的理解：
        metric函数做了下面三件事：
        每一对sample-anchor的匹配度计算: 对应函数中的x矩阵，(num_samples, 9)
        每个sample和当前一整套anchors匹配度的计算: 对应函数中的best, (num_samples, )
        当前数据集和当前整套anchors匹配度的计算，两个指标：
            aat(anchors above threshold): 基于x计算的, sample-anchor匹配度大于阈值的个数 / samples-anchor对总数
            bpr(best possible recall): 基于best计算的, 数据集中和整套anchors匹配度大于阈值的samples个数 / samples总个数

    """
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    # 将每张图片的shape从原图缩放到inp尺寸 # (-1,2)
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True) 
    # augment scale # 为每张图随机指定scale尺度 # (-1, 1)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  
    # wh # 对每张图片随机scale后，将label中的wh恢复到scale后的尺寸上 # (num_samples, 2)
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  

    def metric(k):  # compute metric
        r = wh[:, None] / k[None] # (num_samples,1,2) / (1, 9, 2) = (num_samples, 9, 2)
        # 每个sample的wh和9个anchors的wh比值，以及anchors和sample的wh比值(对于每对sample-anchor有2*2=4个比值）中的最小值
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric # (num_samples, 9) 
        print(x.size())
        # 9个anchors中比值最大的那个比值(宽高匹配度最好的那个)
        best = x.max(1)[0]  # best_x # (num_samples,),  
        # 每个sample和9个anchors最小比值大于阈值的个数的平均值，就是平均每个sample有几个满足大于阈值的匹配anchors # (1)
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold 

        # 当前数据集中和当前anchors匹配度大于阈值的samples个数 / samples总个数 = recall #(1,)
        bpr = (best > 1. / thr).float().mean()  # best possible recall  
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # current anchors # (9,2)
    bpr, aat = metric(anchors)
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # for inference
            check_anchor_order(m) # 检查anchor的顺序和head_layer是否匹配，不匹配就修正
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss # 这里将anchors除以stride, 和model定义时的做法一致
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def kmean_anchors(path='./data/coco.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1. / thr
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, print(f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)
