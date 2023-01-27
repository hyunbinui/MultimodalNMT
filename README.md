![header](https://capsule-render.vercel.app/api?type=transparent&height=200&section=header&text=MultimodalNMT&fontSize=80&fontColor=00925B)

This is the actual implementation of [MultimodalNMT](https://github.com/iacercalixto/MultimodalNMT). Before jumping right into the implementation, note that the original repo is somewhat outdated and based on a \*very\* old version of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).  
  
( You can find the original README.md [here](#original), just in case. )
  
---  

## Introduction  

The original repo is the implementation of **four different multi-modal neural machine translation models** described in the research papers [(1)](http://aclweb.org/anthology/D17-1105) and [(2)](https://aclweb.org/anthology/P/P17/P17-1175.pdf). It provides four different multimodal fusion methods ; IMGd, IMGe, IMGw and src+img. 
-  IMGd / IMGe / IMGw  
  ![스크린샷 2023-01-25 14 22 19](https://user-images.githubusercontent.com/82128320/214990507-edaf9d52-63dc-4e7f-b1c5-6fef7034881a.png)
- src+img  
  ![스크린샷 2023-01-25 14 24 00](https://user-images.githubusercontent.com/82128320/214990934-b55ef6cb-3175-4115-aac3-dd74efe079f1.png)  

For a complete description of the different multi-modal NMT model types, please refer to the papers [(1)](http://aclweb.org/anthology/D17-1105) and [(2)](https://aclweb.org/anthology/P/P17/P17-1175.pdf).
  
They are based on the [Pytorch](https://github.com/pytorch/pytorch) port of [OpenNMT](https://github.com/OpenNMT/OpenNMT), an open-source (MIT) neural machine translation system.  
( But once more, an outdated version of OpenNMT ! You no longer can find this version of onmt module on the official [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) repo. )


## Requirements 
```
python==3.5 or 3.6
```
### *the actual prerequisites*
```
conda install -c soumith pytorch
pip install torchtext --upgrade
pip install -r requirements.actual.txt
pip install pretrainedmodels
conda update pytorch
```
## Implementation
### 0. Clone the Repo
```
! git clone https://github.com/hyunbinui/MultimodalNMT.git
```
### 1. Prepare Dataset
- Download Multi30k dataset [here](https://github.com/multi30k/dataset). Make sure you download raw images. For the test set, I used `test_2017_flickr` set.
- Along with the raw image files, you need respective text files containing the image file names in the training, validation, and test sets. These should be named `train_images.txt`,`val_images.txt`, and `test_images.txt`. ( You might need to adjust the file names accordingly. ) 
  - cf. Text File Annotation
    ```
      1000092795.jpg
      10002456.jpg
      1000268201.jpg
      1000344755.jpg
      1000366164.jpg
      1000523639.jpg
      1000919630.jpg
      10010052.jpg
      1001465944.jpg
      1001545525.jpg
    ```
  - Data Configuration
    ```
    Multi30K
    ├── images
    │      ├── raw_image1.jpg
    │      ├── raw_image2.jpg
    │      └── .....  
    │
    ├── image_split
    │      ├── train_image.txt
    │      ├── val_image.txt
    │      └── test_image.txt
    │
    └── tok #text data → I used the tokenized version.
           ├── train.lc.norm.tok.fr
           ├── val.lc.norm.tok.fr
           ├── test.lc.norm.tok.fr
           ├── train.lc.norm.tok.en
           ├── val.lc.norm.tok.en
           └── test.lc.norm.tok.en
    
    ```
### 2. Extract Image Features
- To extract the image features, run the following script:
  ```
  python extract_image_features.py --gpuid 2 --pretrained_cnn vgg19_bn --splits=train,valid,test --images_path /path/to/flickr30k/images/ --train_fnames /path/to/flickr30k/image_split/train_images.txt --valid_fnames /path/to/flickr30k/image_split/val_images.txt --test_fnames /path/to/flickr30k/image_split/test_images.txt
  ```
- Image features would be extracted in .hdf5 form  
  ![스크린샷 2023-01-25 14 36 17](https://user-images.githubusercontent.com/82128320/214993701-1ac7dcfc-0b4b-4b44-ad68-14dab3921359.png)
### 3. Text Data Preprocessing
- You could tokenize, apply BPE model or do both ! You can easily find tokenized Multi30K text data on our friend Google.
  - To apply BPE model, run the following script:
    ```
    # learn bpe for both tgt & src language
    python learn_bpe.py -i /path/to/flickr30k/tok/train.lc.norm.tok.fr -o /path/to/flickr30k/tok/src.code -s 10000
    python learn_bpe.py -i /path/to/flickr30k/tok/train.lc.norm.tok.en -o /path/to/flickr30k/tok/tgt.code -s 10000
    ```
    ```
    # then apply it → you don't need to apply bpe to test set of tgt language
    python apply_bpe.py  -c /path/to/flickr30k/tok/src.code -i /path/to/flickr30k/tok/train.lc.norm.tok.fr -o /path/to/flickr30k/tok/test.lc.norm.tok.bpe.fr
    python apply_bpe.py  -c /path/to/flickr30k/tok/src.code -i /path/to/flickr30k/tok/val.lc.norm.tok.fr -o /path/to/flickr30k/tok/val.lc.norm.tok.bpe.fr
    python apply_bpe.py  -c /path/to/flickr30k/tok/src.code -i //path/to/flickr30k/tok/test_2017_flickr.lc.norm.tok.fr -o /path/to/flickr30k/tok/test_2017_flickr.lc.norm.tok.bpe.fr

    python apply_bpe.py  -c /path/to/flickr30k/tok/tgt.code -i /path/to/flickr30k/tok/train.lc.norm.tok.en -o /path/to/flickr30k/tok/train.lc.norm.tok.bpe.en
    python apply_bpe.py  -c /path/to/flickr30k/tok/tgt.code -i /path/to/flickr30k/tok/val.lc.norm.tok.en -o /path/to/flickr30k/tok/val.lc.norm.tok.bpe.en
    ```
- Finish preprocessing by running preprocess.py
  ```
  python preprocess.py -train_src /path/to/flickr30k/tok/train.lc.norm.tok.bpe.fr -train_tgt /path/to/flickr30k/tok/train.lc.norm.tok.bpe.en -valid_src /path/to/flickr30k/tok/val.lc.norm.tok.bpe.fr -valid_tgt /path/to/flickr30k/tok/val.lc.norm.tok.bpe.en -save_data /path/to/flickr30k/
  ```
- Processed text data would be saved in .pt form  
  ![스크린샷 2023-01-25 14 46 25](https://user-images.githubusercontent.com/82128320/214996017-84aed508-92ab-4ff4-985d-1ffa033ce8e5.png)
### 4. Train the Model
Run the following script to train the model. Keep in mind that this script expects the path to the training and validation image features, as well as the multi-modal model type (one of `imgd`, `imge`, `imgw`, or `src+img`). If you face torch-related error, try upgrading torch to `torch==1.10.2`. ( or just ask google )
- IMGd
  ```
  python train_mm.py -data /data/IMT/dataset -save_model /path/to/flickr30k/MNMT_model/  -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type imgd
  ```
- IMGe
  ```
  python train_mm.py -data /data/IMT/dataset -save_model /path/to/flickr30k/MNMT_model/  -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type imge
  ```
- IMGw
  ```
  python train_mm.py -data /data/IMT/dataset -save_model /path/to/flickr30k/MNMT_model/  -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type imgw
  ```
- src+img
  ```
  python train_mm.py -data /data/IMT/dataset -save_model /path/to/flickr30k/MNMT_model/  -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection -decoder_type doubly-attentive-rnn --multimodal_model_type src+img
  ```  
### 5. Translate !
- If you just need to translate ( without bleu score or rouge score ... lucky you ! ) , run the following script.
  ```
  python translate_mm.py -src /path/to/flickr30k/tok/test_2017_flickr.lc.norm.tok.bpe.fr -model /path/to/flickr30k/MNMT_model/_acc_74.32_ppl_5.81_e10.pt -path_to_test_img_feats /path/to/flickr30k/flickr30k_test_vgg19_bn_cnn_features.hdf5 -output /home/ubuntu/workspace/230119_imt_hyunbin/MultimodalNMT/results.translations-test2017
  ```  
- If you want to know (or need to know) bleu score or rouge score along with translation, run the following script.
  ```
  python translate_mm.py -src /path/to/flickr30k/tok/test_2017_flickr.lc.norm.tok.bpe.fr -tgt /path/to/flickr30k/tok/test_2017_flickr.lc.norm.tok.en  -model /path/to/flickr30k/MNMT_model/_acc_74.32_ppl_5.81_e10.pt -path_to_test_img_feats /path/to/flickr30k/flickr30k_test_vgg19_bn_cnn_features.hdf5 -output /home/ubuntu/workspace/230119_imt_hyunbin/MultimodalNMT/results/translations-test2017 -report_bleu -report_rouge
  ```  
  - If the script works ( hopefully... ) , you would get the following result.
    ```
      PRED AVG SCORE: -0.2553, PRED PPL: 1.2908
      >> BLEU = 45.66, 74.2/52.7/38.5/28.9 (BP=1.000, ratio=1.011, hyp_len=11496, ref_len=11376)
      >> ROUGE(1/2/3/L/SU4): 74.86/53.80/39.46/73.22/59.02
    ```
  - Elif the scipt does not work and pyrouge error occurs, [this post](https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu) might help you.  
  If you're an impatient person like me, just follow the following process. If this does not help you, reach out to our best friend Google.
    ```
      # Step 1 : Install Pyrouge from source (not from pip)
      git clone https://github.com/bheinzerling/pyrouge
      cd pyrouge
      pip install -e .

      # Step 2 : Install official ROUGE script
      git clone https://github.com/andersjo/pyrouge.git rouge

      # Step 3 : Point Pyrouge to official rouge script -> set !absolute path!
      pyrouge_set_rouge_path ~/pyrouge/rouge/tools/ROUGE-1.5.5/

      # Step 4 : Install libxml parser
      sudo apt-get install libxml-parser-perl

      # Step 5 : Regenerate the Exceptions DB
      cd rouge/tools/ROUGE-1.5.5/data
      rm WordNet-2.0.exc.db
      ./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db

      # Step 6 : Run the tests
      python -m pyrouge.test

      ### If everything went well, you would get the following result ###
      Ran 11 tests in 6.322s
      OK  
    ```  

   
---

  
# Original
## Multi-modal Neural Machine Translation
### OpenNMT-py: Open-Source Neural Machine Translation

[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)

This is the implementation of **four different multi-modal neural machine translation models** described in the research papers [(1)](http://aclweb.org/anthology/D17-1105) and [(2)](https://aclweb.org/anthology/P/P17/P17-1175.pdf).
They are based on the [Pytorch](https://github.com/pytorch/pytorch) port of [OpenNMT](https://github.com/OpenNMT/OpenNMT), an open-source (MIT) neural machine translation system.


Table of Contents
=================
  * [Requirements](#requirements)
  * [Features](#features)
  * [Multi-modal NMT Quickstart](#quickstart)
  * [Citation](#citation)

[Full Documentation](http://opennmt.net/OpenNMT-py/)
 
## Requirements

```
torchtext>=0.2.1
pytorch>=0.2
```

In case one of the two are missing or not up-to-date and assuming you installed pytorch using the conda package manager and torchtext using pip, you might want to run the following:

```bash
conda install -c soumith pytorch
pip install torchtext --upgrade
pip install -r requirements.txt
pip install pretrainedmodels
conda update pytorch
```

## Features

The following OpenNMT features are implemented:

- [data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- [Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [TensorBoard/Crayon logging](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Image-to-text processing](http://opennmt.net/OpenNMT-py/im2text.html)
- [Speech-to-text processing](http://opennmt.net/OpenNMT-py/speech2text.html)

Beta Features (committed):
- multi-GPU
- ["Attention is all you need"](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- Structured attention
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper
- Inference time loss functions.

## Multi-modal NMT Quickstart

### Step 0: Extract the image features for the Multi30k data set.

If you are using image features extracted by someone else, you can skip this step.

We assume you have downloaded the [Multi30k data set](http://www.statmt.org/wmt16/multimodal-task.html) and have the training, validation and test images locally (make sure you download the `test2016` test set). Together with the image files, you need text files with the image file names in the training, validation, and test sets, respectively. These are named `train_images.txt`,`val_images.txt`, and `test_images.txt`, and are part of the original Flickr30k data set. If you download them from the [WMT Multi-modal MT shared task website](http://www.statmt.org/wmt16/multimodal-task.html), you might need to adjust the file names accordingly.

In order to extract the image features, run the following script:

```bash
python extract_image_features.py --gpuid 0 --pretrained_cnn vgg19_bn --splits=train,valid,test --images_path /path/to/flickr30k/images/ --train_fnames /path/to/flickr30k/train_images.txt --valid_fnames /path/to/flickr30k/val_images.txt --test_fnames /path/to/flickr30k/test2016_images.txt
```

This will use GPU 0 to extract features with the pre-trained VGG19 with batch normalisation, for the training, validation and test sets of the Flickr30k. Change the name of the pre-trained CNN to any of the CNNs available under [this repository](https://github.com/Cadene/pretrained-models.pytorch), and the model will automatically use this CNN to extract features. **This script will extract both global and local visual features**.


### Step 1: Preprocess the data

That is the same way as you would do with a text-only NMT model. **Important**: *the preprocessing script only uses the textual portion of the multi-modal machine translation data set*!

In here, we assume you have downloaded the [Multi30k data set](http://www.statmt.org/wmt16/multimodal-task.html) and extracted the sentences in its training, validation and test sets. After pre-processing them (e.g. tokenising, lowercasing, and applying a [BPE model](https://github.com/rsennrich/subword-nmt)), feed the training and validation sets to the `preprocess.py` script, as below.

```bash
python preprocess.py -train_src /path/to/flickr30k/train.norm.tok.lc.bpe10000.en -train_tgt /path/to/flickr30k/train.norm.tok.lc.bpe10000.de -valid_src /path/to/flickr30k/val.norm.tok.lc.bpe10000.en -valid_tgt /path/to/flickr30k/val.norm.tok.lc.bpe10000.de -save_data data/m30k
```


### Step 2: Train the model

To train a multi-modal NMT model, use the `train_mm.py` script. In addition to the parameters accepted by the standard `train.py` (that trains a text-only NMT model), this script expects the path to the training and validation image features, as well as the multi-modal model type (one of `imgd`, `imge`, `imgw`, or `src+img`).

For a complete description of the different multi-modal NMT model types, please refer to the papers where they are described [(1)](http://aclweb.org/anthology/D17-1105) and [(2)](https://aclweb.org/anthology/P/P17/P17-1175.pdf).

```bash
python train_mm.py -data data/m30k -save_model model_snapshots/IMGD_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/features/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/features/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type imgd
```

In case you want to continue training from a previous checkpoint, simply run (for example):

```bash
MODEL_SNAPSHOT=IMGD_ADAM_acc_60.79_ppl_8.38_e4.pt
python train_mm.py -data data/m30k -save_model model_snapshots/IMGD_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/features/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/features/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type imgd -train_from model_snapshots/${MODEL_SNAPSHOT}
```

As an example, if you wish to train a doubly-attentive NMT model (referred to as `src+img`), try the following command:

```bash
python train_mm.py -data data/m30k -save_model model_snapshots/NMT-src-img_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/features/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/features/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --decoder_type doubly-attentive-rnn --multimodal_model_type src+img
```


### Step 3: Translate new sentences

To translate a new test set, simply use `translate_mm.py` similarly as you would use the original `translate.py` script, with the addition of the path to the file containing the test image features. In the example below, we translate the Multi30k test set used in the 2016 run of the WMT Multi-modal MT Shared Task.

```bash
MODEL_SNAPSHOT=IMGD_ADAM_acc_60.79_ppl_8.38_e4.pt
python translate_mm.py -src ~/exp/opennmt_imgd/data_multi30k/test2016.norm.tok.lc.bpe10000.en -model model_snapshots/${MODEL_SNAPSHOT} -path_to_test_img_feats ~/resources/multi30k/features/flickr30k_test_vgg19_bn_cnn_features.hdf5 -output model_snapshots/${MODEL_SNAPSHOT}.translations-test2016
```

## Citation

If you use the multi-modal NMT models in this repository, please consider citing the research papers where they are described [(1)](http://aclweb.org/anthology/D17-1105) and [(2)](https://aclweb.org/anthology/P/P17/P17-1175.pdf):

```
@InProceedings{CalixtoLiu2017EMNLP,
  Title                    = {{Incorporating Global Visual Features into Attention-Based Neural Machine Translation}},
  Author                   = {Iacer Calixto and Qun Liu},
  Booktitle                = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  Year                     = {2017},
  Address                  = {Copenhagen, Denmark},
  Url                      = {http://aclweb.org/anthology/D17-1105}
}
```

```
@InProceedings{CalixtoLiuCampbell2017ACL,
  author    = {Calixto, Iacer  and  Liu, Qun  and  Campbell, Nick},
  title     = {{Doubly-Attentive Decoder for Multi-modal Neural Machine Translation}},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {1913--1924},
  url       = {http://aclweb.org/anthology/P17-1175}
}
```

If you use OpenNMT, please cite as below.

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
