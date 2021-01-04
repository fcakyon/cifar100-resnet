# ResNet Implementation for CIFAR100 in Pytorch
[Torchvision model zoo](https://github.com/pytorch/vision/tree/master/torchvision/models) provides number of implementations of various state-of-the-art architectures, however, most of them are defined and implemented for ImageNet. Usage of these backbones with small input sizes (as in CIFAR) is not trivial.

This repo provides training and evaluation scripts for CIFAR100 with ResNet backbones. 60% accuracy can be obtained with default training parameters.

Check [the notebook](<https://github.com/fcakyon/cifar100-resnet/tree/main/notebook/CIFAR-100 Resnet.ipynb>) for a demo of the submodules used in this repo.

## Usage
- Clone:
```bash
git clone https://github.com/fcakyon/cifar100-resnet.git
```

- Prepare conda environment:
```bash
conda env create -f environment.yml
```

```bash
conda activate cifarresnet
```

- Train:
```bash
python main.py --arch resnet32 --save-dir checkpoints/
```

- Evaluate:
```bash
python main.py --evaluate checkpoints/resnet32_final.th
```



## Related Projects
[pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)