# AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors
The official PyTorch implementation for "*AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors*", ICLR 2025

**Authors: [Ruoxuan Feng](https://xxuan01.github.io/), Jiangyu Hu, [Wenke Xia](https://xwinks.github.io/), Tianci Gao, Ao Shen, [Yuhao Sun](https://scholar.google.com.hk/citations?user=ShKpk00AAAAJ), [Bin Fang](https://scholar.google.com/citations?hl=zh-CN&user=5G47IcIAAAAJ), [Di Hu](https://dtaoo.github.io/)**

**Accepted by: International Conference on Learning Representations (ICLR 2025)**

**Resources:[[Project Page]()],[[Arxiv]()],[[Checkpoints]()],[[Dataset]()]**

If you have any questions, please open an issue or send an email to [fengruoxuan@ruc.edu.cn](mailto:fengruoxuan@ruc.edu.cn).

------

## Introduction

<div align="center">    
<img src="assest/intro.png" width = "90%" />
</div>

Tactile perception is crucial for humans to perceive the physical world. Over the years, various **visuo-tactile sensors** have been designed to endow robots with human-like tactile perception abilities. However, the **low standardization** of visuo-tactile sensors has hindered the development of a powerful tactile perception system. In this work, we present [**TacQuad**](), an **aligned multi-modal multi-sensor tactile dataset** that enables the explicit integration of sensors. Building on this foundation and other open-sourced tactile datasets, we propose learning unified representations from **both static and dynamic perspectives** to accommodate a range of tasks. We introduce **AnyTouch**, a **unified static-dynamic multi-sensor** tactile representation learning framework with a **multi-level** architecture, enabling comprehensive static and **real-world** dynamic tactile perception.

## TacQuad Dataset

<div align="center">    
<img src="assest/dataset.png" width = "90%"/>
</div>

TacQuad is an aligned multi-modal multi-sensor tactile dataset collected from 4 types of visuo-tactile sensors (GelSight Mini, DIGIT, DuraGel and Tac3D). It offers a more comprehensive solution to the low standardization of visuo-tactile sensors by providing multi-sensor aligned data with text and visual images. This explicitly enables models to learn semantic-level tactile attributes and sensor-agnostic features to form a unified multi-sensor representation space through data-driven approaches. This dataset includes two subsets of paired data with different levels of alignment:

- **Fine-grained spatio-temporal aligned data:** This portion of the data was collected by pressing the same location of the same object at the same speed with the four sensors. It contains a total of 17,524 contact frames from 25 objects, which can be used for fine-grained tasks such as cross-sensor generation.
- **Coarse-grained spatial aligned data:** This portion of the data was collected by hand, with the four sensors pressing the same location on the same object, although temporal alignment is not guaranteed. It contains 55,082 contact frames from 99 objects, including both indoor and outdoor scenes, which can be used for cross-sensor matching task.

We also use GPT-4o to generate or expand the text modality for several open-sourced tactile datasets. The TacQuad dataset and text prompt for other datasets are hosted on [HuggingFace](). 

## AnyTouch Model

<div align="center">    
<img src="assest/model.png" width = "90%"/>
</div>

AnyTouch is a unified static-dynamic multi-sensor tactile representation learning framework which integrates the input format of tactile images and videos. It learns both fine-grained pixel-level details for refined tasks and semantic-level sensor-agnostic features for understanding properties and building unified space by a multi-level structure. 

The checkpoint for AnyTouch is provided below:

|          |                        Training Data                         |     TAG (M/R/H)*      | Feel (Grasp) | OF 1.0 | OF 2.0 |              |
| -------- | :----------------------------------------------------------: | :-------------------: | :----------: | :----: | :----: | :----------: |
| AnyTouch | TAG, VisGel, Cloth, TVL, SSVTP,<br>YCB-Slide, OF Real, Octopi, TacQuad | 80.82 / 86.74 / 94.68 |    80.53     | 49.62  | 76.02  | [Download]() |

*M: Material   R:Roughness   H:Hardness

## Setup

This code is tested in Ubuntu 20.04, PyTorch 2.1.0, CUDA 11.8

**Install the requirements**

```
# Optionally create a conda environment
conda create -n anytouch python=3.9
conda activate anytouch
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

```



## Run







## Citation
