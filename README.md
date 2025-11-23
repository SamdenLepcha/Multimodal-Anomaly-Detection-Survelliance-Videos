# Multimodal Anomaly Detection on Surveillance Videos

### Setup
It is recommended to use a Linux machine compatible with CUDA-compatible GPUs; however, it is also compatible with Windows systems and GPUs, as well as macOS for inference.

Clone the repo with:
```
git clone https://github.com/SamdenLepcha/Multimodal-Anomaly-Detection-Survelliance-Videos.git
```

### Virtual Environment
Please download and install Python on your local system first. The environment can then be installed and activated with:
```
python -m venv venv

# Windows (Command Prompt or PowerShell)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Data
Please download the official data sources from these locations:
- UCF-Crime: [Official Website](https://www.crcv.ucf.edu/projects/real-world/)
- UCA: [GitHub Repo](https://github.com/Xuange923/Surveillance-Video-Understanding)

### Pretrained Models
We employ pre-trained models for each segment, which can be either downloaded or contain instructions to download them:
| Model     | Link                                                                                               |
| ----------- | -------------------------------------------------------------------------------------------------- |
| I3D Feature Extractor   | [Github](https://github.com/SamdenLepcha/Multimodal-Anomaly-Detection-Survelliance-Videos/tree/main/I3D_Feature_Extraction_resnet) |
| SwinBERT | [GitHub](https://github.com/SamdenLepcha/Multimodal-Anomaly-Detection-Survelliance-Videos/tree/main/SwinBERT) |
| TEVAD (Official) | [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbGJEekE5RDhWa2hvTzhkY3ZKTmFBTWtrNWJiZ0E%5FZT1FaDJMQ0I&id=2159F1430FCCC356%21538528&cid=2159F1430FCCC356&sb=name&sd=1)|
| Enhanced TEVAD | 🤗 [Hugging Face](https://huggingface.co/bayesicallysam/EnhancedTEVAD)|

#### File Structure
```
|-- README.md
|-- dataprep  # data prep notebooks
|   |-- DataPrep.ipynb
|   |-- EDA.ipynb
|-- requirements.txt
|-- I3D_Feature_Extraction_resnet  # Visual Feature (I3D) Generation Module
|   |-- output
|   |-- pretrained # Place Pretrained Models here
|   |   |-- i3d_baseline_32x2_IN_pretrain_400k.pkl
|   |   |-- i3d_r50_kinetics.pth
|   |   |-- i3d_r50_kinetics.pth.keymap
|   |-- utils
|   |-- requirements.txt
|   |-- main.py
|   |-- resnet.py
|   |-- extract_features_org.py
|   |-- extract_features.py
|-- SwinBERT  # Caption/Embedding Generation Module
|   |-- models
|   |   |-- captioning
|   |   |   |--bert-base-uncased
|   |   |-- table1
|   |   |-- vatex # Original Model
|   |   |   |--best-checkpoint
|   |   |   |--log
|   |   |   |--tokenizer
|   |   |-- uca_finetune # Finetuned on UCA
|   |   |   |--best-checkpoint
|   |   |   |--log
|   |   |   |--tokenizer
|   |   |-- video_swin_transformer
|   |   |   |-- swin_base_patch244_window877_kinetics400_22k.pth
|   |   |   |-- swin_base_patch244_window877_kinetics600_22k.pth
|   |-- src
|   |-- datasets # External Output location
|   |-- outputs # Local Folder Output Location
|   |-- prepro # Local Folder Output Location
|-- TEVAD  # Multimodal Framework
|   |-- ckpt # External Output location
|   |   |-- my_best
|   |   |   |-- best-checkpoint.pkl # Trained Model
|   |-- list
|   |-- output
|   |-- results
|-- visio_guard  # Flask app
|   |-- models
|   |   |-- best-model.pkl
|   |   |-- model.py
|   |   |-- tevad_runner.py
|   |-- static
|   |   |-- outputs # Visual and Embeddings for the uploaded video gets created here
|   |   |-- uploads # Uploaded Video gets saved here
|   |-- templates # Rendered Template
|   |   |-- index.html 
|   |   |-- report.html 
|   |-- utils
|   |   |-- captioner.py
|   |   |-- file_utils.py
|   |   |-- plot_utils.py
|   |   |-- text_utils.py
|   |   |-- video_utils.py
|   |-- app.py # Main file to start web server
```

#### Start App
```
cd visio_guard

python app.py
```

# Acknowledgements
This code is based on the following repositories. Special thanks to these authors for their great work and simple setup instructions. Please feel free to check them out:
- TEVAD [Official](https://github.com/coranholmes/TEVAD)
- SwinBERT [Official](https://github.com/microsoft/SwinBERT) and [Forked](https://github.com/coranholmes/SwinBERT)
- I3D Feature Extraction: [Paper Implementation](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet)
