![AviBert-cover](https://www.aeroturk.info/wp-content/uploads/2019/11/Aircraft-Systems.jpg)
<h2 align="center">AviBert: on Classifying the news about Aircraft</h2>
<p align="center">
  Developed by <a href="https://github.com/ByUnal"> M.Cihat Unal </a> 
</p>

## Overview

This repository focuses on Aircraft, and we work towards developing an Aircraft-specific classification model on a multi-class development set by using BERT and its lightweight and heavyweight variants. Besides, introduces a pipeline that comprises data collection, data tagging and model training. 
Overall, since data and targets are unique, the presented model in this study is also a groundbreaker. Details of the dataset can be investigated further, and the results are compared by using macro-f1 and accuracy scores between models.


## Setup
Install the requirements. I've added torch to requirements.txt, but you can prefer to install by yourself according to different cuda version and resources.
```commandline
pip install -r requirements.txt
```

## Run the Code
I've concluded hyperparameter tuning by using optuna, and therefore main.py fixed accordingly. Also, you can train standalone model by using *train_loop()*

## Results
The results that we obtained our experiments as below:
![plot](./results/acc-f1_scores.png)

You can also see the best parameters for the models after hyperparameter optimization in *results/params.txt*
## Acknowledgement
Currently, I've prepared the paper of this project besides including data collection steps. However, we're doing an additional novel experiments on this topic.
So, paper link/details will be shared as soon as the paper is published. 
