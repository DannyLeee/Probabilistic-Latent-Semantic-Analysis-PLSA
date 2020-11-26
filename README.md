## Introduction
* This is the Information Retrieval HW2
* Using PLSA to compute the relation between given querys and documents

## Usage
```
code.py [-h] [-from_scratch FROM_SCRATCH] [-train_from TRAIN_FROM]
               -A alpha -B beta -K K -step STEP

optional arguments:
  -h, --help            show this help message and exit
  -from_scratch FROM_SCRATCH
                        default load computed tf, BG and voc (default: 0)
  -train_from TRAIN_FROM
                        load the computed model to continue (default: 0)
  -A alpha              (default: 0.7)
  -B beta               (default: 0.2)
  -K K                  K latent topic (default: 16)
  -step STEP            EM iterattion time (default: 30)
```

## Approach
* lexicon use top frequent 10,000 word from all documents
* latent topic number <img src="https://latex.codecogs.com/gif.latex?K"/>: 16
* <img src="https://latex.codecogs.com/gif.latex?\alpha=0.7"/>
* <img src="https://latex.codecogs.com/gif.latex?\beta=0.2"/>
* EM iteration step: 30