# Online 3D Gaussian Splatting with Novel View Selection

Official repository for **Online 3D Gaussian Splatting Modeling with Novel View Selection** (**arXiv:2508.14014**).  
Paper: [arXiv](https://arxiv.org/abs/2508.14014)

This project builds an online 3D Gaussian Splatting model from an RGB image stream with:
- Tracking / pose estimation: DROID-SLAM
- Online depth estimation: MVSFormer
- Mapping / optimization: Online 3DGS with Novel View Selection (NVS)

## Installation

### 1) Create environment

```bash
conda create -n mvs_droidslam python=3.8 -y
conda activate mvs_droidslam

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 \
  -c pytorch -c nvidia
```

### 2) Clone repository

```bash
git clone --recursive https://github.com/lbg030/MVS-Droid_SLAM.git
cd MVS-Droid_SLAM
```

### 3) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4) Build CUDA extensions

```bash
python setup.py install
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## Download Pretrained Weights

### A) DROID-SLAM weight

```bash
pip install gdown
mkdir -p weights
gdown --fuzzy "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing" \
  -O weights/droid.pth
```

### B) MVSFormer weight

Download from:
- https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaDJWa1VMbWtpcVByeUhfVGwyUFVTNklzODMxP2U9QmdDdU9Z&cid=8F2A92E64291951D&id=8F2A92E64291951D%216051&parId=8F2A92E64291951D%216049&o=OneUp

Then place it at:

```bash
mkdir -p weights
# move downloaded checkpoint to:
# weights/mvsformer.pth
```

## Dataset Config



## Quickstart

```bash
python demo.py --config <path_to_your_scene_config.yaml>
```


## Outputs


## Citation

If you find our work useful, please consider citing:

### MVS-GS
```bibtex
@article{lee2025mvsgs,
  title={MVS-GS: High-Quality 3D Gaussian Splatting Mapping via Online Multi-View Stereo},
  author={Lee, Byeonggwon and Park, Junkyu and Truong Giang, Khang and Jo, Sungho and Song, Soohwan},
  journal={IEEE Access},
  volume={13},
  pages={1--13},
  year={2025},
  doi={10.1109/ACCESS.2025.3583156}
}


@misc{lee2025online3dgaussiansplatting,
      title={Online 3D Gaussian Splatting Modeling with Novel View Selection},
      author={Byeonggwon Lee and Junkyu Park and Khang Truong Giang and Soohwan Song},
      year={2025},
      eprint={2508.14014},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14014},
}
```

## License

See [LICENSE](LICENSE).
