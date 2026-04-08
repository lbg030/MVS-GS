# MVS-GS / Online 3D Gaussian Splatting with Novel View Selection

🚀 **Official repository for:**

* *Online 3D Gaussian Splatting Modeling with Novel View Selection* (IJCAI 2025)
* *MVS-GS: High-Quality 3D Gaussian Splatting Mapping via Online Multi-View Stereo* (IEEE Access 2025)

👉 [Paper (arXiv)](https://arxiv.org/abs/2508.14014)

---

## 🔥 Overview

We present an **online 3D Gaussian Splatting framework** that reconstructs a scene from an RGB image stream.

The system integrates **SLAM, multi-view stereo, and neural rendering** into a unified pipeline:

* Informative view selection for efficient reconstruction
* Multi-view stereo for geometric consistency
* Online mapping and optimization

---

## 🧩 Pipeline

The framework consists of three main components:

1. **Tracking / Pose Estimation**
   Camera poses are estimated from RGB inputs using DROID-SLAM

2. **Online Depth Estimation**
   Multi-view consistent depth is predicted using MVSFormer

3. **Mapping / Optimization**
   Online 3D Gaussian Splatting with:

   * MVS-based geometric initialization
   * Novel view selection for efficient updates

---

## ⚙️ Installation

### 1) Create environment

```bash
conda create -n mvs_droidslam python=3.8 -y
conda activate mvs_droidslam

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 \
  -c pytorch -c nvidia
```

---

### 2) Clone repository

```bash
git clone --recursive https://github.com/lbg030/MVS-Droid_SLAM.git
cd MVS-Droid_SLAM
```

---

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4) Build CUDA extensions

```bash
python setup.py install
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

---

## 📦 Pretrained Weights

### DROID-SLAM

```bash
pip install gdown
mkdir -p weights
gdown --fuzzy "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing" \
  -O weights/droid.pth
```

---

### MVSFormer

Download from:
https://onedrive.live.com/...

```bash
mkdir -p weights
# place checkpoint at:
# weights/mvsformer.pth
```

---

## 🚀 Quickstart

```bash
python demo.py --config <path_to_config.yaml>
```

---

## 📂 Outputs

The system produces:

* Reconstructed 3D Gaussian model
* Camera trajectory
* Depth and intermediate mapping results

---

## 📄 Citation

If you find our work useful, please consider citing:

### MVS-GS (IEEE Access)

```bibtex
@article{lee2025mvs,
  title={Mvs-gs: High-quality 3d gaussian splatting mapping via online multi-view stereo},
  author={Lee, Byeonggwon and Park, Junkyu and Giang, Khang Truong and Jo, Sungho and Song, Soohwan},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```

---

### Online 3D Gaussian Splatting with Novel View Selection (IJCAI2025)

```bibtex
@article{lee2025online,
  title={Online 3D Gaussian splatting modeling with novel view selection},
  author={Lee, Byeonggwon and Park, Junkyu and Giang, Khang Truong and Song, Soohwan},
  journal={arXiv preprint arXiv:2508.14014},
  year={2025}
}
```

---

## 📜 License

See [LICENSE](LICENSE)
