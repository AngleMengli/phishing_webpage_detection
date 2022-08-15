# Phishing Webpage Detection Based on Global and Local Visual Similarity
This repository is the implementation for our paper :
Phishing Webpage Detection Based on Global and Local Visual Similarity(Mengli Wang;Lipeng Song;Luyang Li;Yuhui Zhu;Jing Li)
# Datasets
The dataset is from the paper "VisualPhishNet: Zero-Day Phishing Website Detection by Visual Similarity."
# Requirments
* python 3+
* pytorch 0.4+
* numpy
* datetime
# Train the model
If you want to train the NTS-Net, just run `python train.py`. You may need to change the configurations in `config.py`. The parameter `PROPOSAL_NUM` is `p` in the original paper and the parameter `CAT_NUM` is `k` in the original paper. During training, the log file and checkpoint file will be saved in `save_dir` directory. You can change the parameter `resume` to choose the checkpoint model to resume.

