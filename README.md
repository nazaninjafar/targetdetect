# TargetDetect- Target Span Detection for Implicit Harmful Content

Identifying targets of hate speech is vital for understanding and mitigating harmful content on online platforms, especially when implicit language is used. This project focuses on detecting implied targets of hate speech to enhance the detection of offensive posts. We define a new task of identifying these targets in three key datasets: SBIC, DynaHate, and IHC, and introduce a merged collection called Implicit-Target-Span. Using an innovative pooling method with human annotations and Large Language Models (LLMs), this collection offers a challenging test bed for target span detection methods.


## Table of Contents

- [Dataset](#Dataset)
- [Training Mutas](#Training)
<!-- - [Citation](#Citation) -->

## Dataset

The generated dataset using our pooling based approach is provided in the `data/targeted` directory for `SBIC`, `DynaHate`, and `IHC` datasets. 

## Training

To train the model from scratch and reproduce our results, use the following script: 
```bash
python mutas/train.py  --dataset [DATASET] --model [MODEL] --test_dataset [TEST DATASET]
``` 
For `dataset` and `test_dataset` you can choose from `sbic`, `dynahate` and `implicit-hate-corpus`. 

For `model` we used `roberta-large`, `bert-base-uncased ` and `GroNLP/hateBERT`.  

<!-- ## Citation 

If you use our data or code, please cite it as follows: 

@article{jafari2024,
title={Target Span Detection for Implicit Harmful Content},
author={Your First Name, Your Last Name and Co-author First Name, Co-author Last Name},
journal={Journal Name},
volume={XX},
number={XX},
pages={XX-XX},
year={2024},
publisher={Publisher Name}
} -->

