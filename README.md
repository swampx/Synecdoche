# Synecdoche: Efficient and Accurate In-Network Traffic Classification

[![INFOCOM 2026](https://img.shields.io/badge/INFOCOM-2026-blue.svg)](https://infocom2026.ieee-infocom.org/) [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](https://claude.xiaoai.shop/chat/LICENSE)

Official implementation of **"Synecdoche: Efficient and Accurate In-Network Traffic Classification via Direct Packet Sequential Pattern Matching"**, accepted at IEEE INFOCOM 2026.

## 📖 Overview

Synecdoche is a novel traffic classification framework that bridges the accuracy-efficiency gap on programmable data planes through direct packet sequential pattern matching. Our key innovation is leveraging **Key Segments** - discriminative packet subsequences that encapsulate the essence of traffic patterns - enabling high-accuracy classification with minimal hardware resources.

![Synecdoche](D:\code\Synecdoche\README.assets\Synecdoche.png)

### Key Features

- **High Accuracy**: Achieves up to 26.4% F1-score improvement over statistical methods and 18.3% over online deep learning approaches
- **Resource Efficient**: 79.2% reduction in SRAM usage and 13.0% lower latency compared to existing methods
- **Automated Discovery**: Deep learning-powered Key Segment extraction using CNN + Grad-CAM
- **Line-Rate Deployment**: Direct pattern matching on P4 programmable switches for real-time classification



## 🚀 Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Synecdoche.git
cd Synecdoche
```

1. Install dependencies:

```bash
conda install tensorflow
pip install -r requirements.txt
```



## 📊 Usage

### Step 1: Configure Parameters

Edit `config.yaml` to customize dataset paths and hyperparameters:

```yaml
dataset:
  name: 'your_dataset_name'
  pcap_folder: 'dataset_pcap/'
  txt_folder: 'dataset_txt/'
  num_classes: 16

preprocess:
  sample_limit: 10000      # Max samples per class
  max_pkt_number: 32       # Flow length (number of packets)
  min_pkt_number: 5        # Minimum flow length
  split_ratio: 12          # Train:valid:test = 10:1:1

cnn_training:
  embed_dimension: 128     # Embedding dimension
  filter: 128              # CNN filter size
  epochs: 60               # Training epochs
  batch_size: 32			

rules_generating:
  max_length: 4            # Max Key Segment length
  eps: 1                   # DBSCAN epsilon
  score: 1.5               # Score threshold for filtering
```

### Step 2: Data Preparation

**2.1 Extract Features from PCAP Files**

Place your raw PCAP files in `dataset_pcap/` with subdirectories for each class:

```
dataset_pcap/
├── class1/
│   ├── sample1.pcap
│   └── sample2.pcap
├── class2/
│   └── ...
```

Run feature extraction:

```bash
python data_preprocess/pcap2txt.py
```

This converts PCAP files to text format with packet size sequences (with direction information).

**2.2 Split Dataset**

Split the extracted features into train/valid/test sets:

```bash
python data_preprocess/split_dataset.py
```

This creates `train/`, `valid/`, and `test/` subdirectories in `dataset_txt/`.

### Step 3: Offline Discovery

**Train CNN Model & Extract Key Segments**

Run the complete offline discovery pipeline:

```bash
python offline_discovery/train_model.py
```

This script will:

1. Train a 1D-CNN model for traffic classification
2. Apply Grad-CAM to extract candidate segments
3. Cluster segments using DBSCAN
4. Filter segments based on discriminative scores
5. Save Key Segments to `key_segments/` in JSON format
6. Save the trained model to `saved_model/`

### Step 4: Online Matching Simulation

This step provides both Python-based simulation and P4 hardware deployment options.

#### 4.1 Python Simulation

For testing and validation, we provide a Python-based simulation of the P4 data plane behavior.

**Train Backup Decision Tree:**

bash

```bash
python online_matching/python_simulation/train_rf.py
```

This trains a decision tree classifier as a backup mechanism for flows that don't match any Key Segment.

**Simulate P4 Matching:**

```bash
python online_matching/python_simulation/match_p4.py
```

This simulates the complete P4 data plane matching behavior and outputs comprehensive evaluation metrics:

#### 4.2 P4 Hardware Deployment

For line-rate deployment on Intel Tofino switches, we provide P4 implementation in `online_matching/p4/`.

**Note:** The P4 code is currently being finalized and will be released soon. 

## 📈 Experimental Results

Our experiments on public datasets demonstrate:

| Dataset           | F1-Score |
| ----------------- | -------- |
| Bot-IoT           | 95.9%    |
| ToN-IoT           | 79.3%    |
| CipherSpectrum-10 | 91.5%    |
| VisQUIC-10        | 89.7%    |

## 🔧 Configuration

### Key  Parameters

- `max_length`: Maximum length of Key Segments (default: 4)
  - Shorter segments: faster matching, may sacrifice accuracy
  - Longer segments: higher accuracy, more resources
- `score`: Discriminative score threshold (default: 1.5-3.0)
  - Higher values: fewer but higher-quality segments
  - Lower values: more segments, higher coverage

## 🎯 Supported Datasets

The framework has been tested on:

1. **IoT Security Detection**:
   - [Bot-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset)
   - [ToN-IoT]((https://research.unsw.edu.au/projects/toniot-datasets))
2. **Application Classification**:
   - [CipherSpectrum](https://cgi.cse.unsw.edu.au/~cspectrum/)
   - [VisQUIC](https://github.com/robshahla/VisQUIC)

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{synecdoche2026,
  title={Synecdoche: Efficient and Accurate In-Network Traffic Classification via Direct Packet Sequential Pattern Matching},
  author={[Xiao Minyuan, Li YunChun, Zhao Yuchen, Guan Tong, Xia Mingyuan, Li Wei]},
  booktitle={IEEE International Conference on Computer Communications (INFOCOM)},
  year={2026}
}
```

## 🙏 Acknowledgments

- Thanks to the authors of Grad-CAM, FS-Net, NetBeacon, and Brain-on-Switch for their pioneering work
- Dataset providers: UNSW (Bot-IoT, ToN-IoT), UMass (CipherSpectrum), BIU (VisQUIC)

## 📧 Contact

For questions and feedback, please contact:

- Email: [xiaominyuan@buaa.edu.cn]

