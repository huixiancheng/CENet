# SENET: TOWARD SIMPLE AND EFFICIENT LIDAR SEMANTIC SEGMENTATION FOR AUTONOMOUS DRIVING

## Abstract：
Accurate and fast scene understanding is one of the challenging task for autonomous driving, which requires to take full advantage of LiDAR point clouds for semantic segmentation. In this paper, we present a simple and efficient image-based semantic segmentation network, named SENet. In order to improve the descriptive power of learned features and reduce the computational as well as time complexity, our SENet integrates the convolution with larger kernel size instead of MLP, carefully-selected activation functions, and multiple auxiliary segmentation heads with corresponding loss functions into architecture. Quantitative and qualitative experiments conducted on publicly available benchmarks, SemanticKITTI and SemanticPOSS, demonstrate that our pipeline achieves much better mIoU and inference performance compared with state-of-the-art models.

## Prepare
Download SemanticKITTI from [official web](http://www.semantic-kitti.org/dataset.html).

Download SemanticPOSS from [official web](http://www.poss.pku.edu.cn./download.html).

## Usage：
### Train：
- SemanticKITTI:

    `python train.py -d /your_dataset -ac config/arch/senet-512.yml -n senet-512`

- SemanticPOSS:

    `python train_poss.py -d /your_dataset -ac config/arch/poss.yml -n res`

### Infer and Eval：
- SemanticKITTI:

    `python infer.py -d /your_dataset -l /your_predictions_path -m trained_model -s valid/test`
    
    Eval for valid sequences:

    `python evaluate_iou.py -d /your_dataset -p /your_predictions_path`

    For test  sequences, need to upload to [CodaLab](https://competitions.codalab.org/competitions/20331#participate) pages.

- SemanticPOSS:

    `python infer_poss.py -d /your_dataset -l /your_predictions_path -m trained_model`

    This will generate both predictions and mIoU results.

### Visualize Example:


- Visualize GT:

  `python visualize.py -w kitti/poss -d /your_dataset -s sequences`

- Visualize Predictions:

  `python visualize.py -w kitti/poss -p /your_predictions -s sequences`


### Pretrained Model and Logs:
Coming soon.

### TODO List

- [ ] Release Pretrained Model and Logs.
- [ ] Try TensorRT acceleration.
- [ ] To make NLA adaptation framework, See [here](https://github.com/huixiancheng/SENet/blob/57d3e07777099c805fa27ceda68e359b2b7ae12d/modules/user.py#L178-L194).

### Acknowledgments：
Code framework derived from from [SalsaNext](https://github.com/Halmstad-University/SalsaNext). Models are heavily based on [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI). Part of code from [SqueezeSegV3](https://github.com/chenfengxu714/SqueezeSegV3). Thanks to their open source code, and also to [Ph.D. Zhao](https://github.com/placeforyiming) for some helpful discussions.

### Citation：
Soon
