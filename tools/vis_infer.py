# This script is used to visualize the prediction results
from mmdet.apis import DetInferencer


def vis_infer(
    checkpoints="D:/history/论文1/补充实验/mask/best_coco_segm_mAP_epoch_23.pth",
    config='D:/history/论文1/补充实验/mask/mask.py',
    data_dir='D:/project/HySaM/dataset/uw/test',
    output_dir='D:/history/论文1/补充实验/mask'
):
    """
    Function to run the DetInferencer for visual inference with default parameters.

    Args:
    checkpoints (str): Path to the model checkpoint (default: "./pretrain/mus.pth").
    config (str): Path to the configuration file (default: './project/our/configs/.py').
    data_dir (str): Path to the directory containing the test uw (default: './uw/test/').
    output_dir (str): Path to the output directory where results will be saved (default: './uw/vis/test').
    """
    # Initialize the DetInferencer
    inferencer = DetInferencer(model=config, weights=checkpoints)
    
    # Perform inference and save the output
    inferencer(data_dir, out_dir=output_dir)


if __name__ == "__main__":
    vis_infer()
