"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-08-03 06:45
 * Filename      : gen_dataset_fp_list.py
 * Description   : 
"""
import os

if __name__ == "__main__":
    with open("FMS_data/val.txt", "w")as wr:
        src_dir = "./FMS_data/images/fms-noframeRot/val2017/"
        save_prefix = "./images/fms-noframeRot/val2017/" 
    # with open("FMS_data/train.txt", "w")as wr:
        # src_dir = "./FMS_data/images/fms-noframeRot/train2017/"
        # save_prefix = "./images/fms-noframeRot/train2017/" 
        imgs = os.listdir(src_dir)
        for img in imgs:
            wr.write(os.path.join(save_prefix, img) + '\n')
