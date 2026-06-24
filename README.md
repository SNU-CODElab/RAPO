# RAPO: Retrieval-Augmented Phase Ordering

This repository contains the official implementation of RAPO: Retrieval-Augmented Phase Ordering, published at the 27th ACM SIGPLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems (LCTES 2026).

RAPO performs compiler phase ordering without running an online pass search. It embeds LLVM IR programs, retrieves optimization sequences from nearby program clusters, and applies the retrieved sequences to unseen programs.


## Overview

RAPO performs compiler phase ordering without running an online optimization search for each input program. It represents LLVM IR programs as fixed-length embeddings, groups similar programs into clusters, and reuses optimization sequences obtained from representative programs.

The overall workflow consists of four stages:

1. Train a WordPiece tokenizer and an IR-BERT model on LLVM IR.
2. Extract a fixed-length embedding for each LLVM IR program.
3. Cluster the embeddings and select representative programs for each cluster.
4. Retrieve and apply optimization sequences from the nearest clusters to unseen programs.

```text
LLVM IR
   │
   ▼
IR Tokenizer
   │
   ▼
IR-BERT
   │
   ▼
IR Embeddings
   │
   ▼
K-means Clustering
   │
   ▼
Representative Optimization Sequences
   │
   ▼
Retrieval-Based Phase Ordering
```


## Repository Structure
```text
├── IR-BERT
│   ├── ir_tokenizer.py
│   ├── ir_bert_training.py
│   └── embedding_csv.py
│
└── RAPO
    ├── apply_ppo
    │   ├── generate_sequences.py
    │   └── analyze_instcount.py
    │
    ├── clustering
    │   └── kmeans_clustering.py
    │
    └── rapo
        ├── FULL_ZS_pipeline.py
        └── FULL_TOP5_pipeline.py
```

## Expected Dataset Layout

A typical training dataset layout is:

```text
dataset/
├── Source_1/
│   └── llvm_O0.ll
├── Source_2/
│   └── llvm_O0.ll
├── Source_3/
│   └── llvm_O0.ll
└── ...
```

The tokenizer training, IR-BERT training, and embedding extraction scripts can read these files using a pattern such as:

```text
/path/to/dataset/Source_*/llvm_O0.ll
```

The RAPO evaluation pipelines expect each program to be stored in a separate directory:

```text
evaluation_dataset/
├── Source_200001/
│   ├── Source.c
│   ├── llvm_O0.ll
│   ├── llvm_OZ.ll
│   └── input.bc
├── Source_200002/
│   ├── Source.c
│   ├── llvm_O0.ll
│   ├── llvm_OZ.ll
│   └── input.bc
└── ...
```

## Prerequisites

The experiments in the paper were conducted using:

* Linux
* Clang/LLVM 10.0.0
* Python 3.8
* CompilerGym 0.2.5 (https://github.com/facebookresearch/compilergym)
* SLTrans Dataset (https://huggingface.com/datasets/UKPLab/SLTrans)

## Paper

**RAPO: Retrieval-Augmented Phase Ordering**

Jinwook Yang, Junghyun Lee, Yeonsun Hong, and Hyojin Sung
27th ACM SIGPLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems (**LCTES 2026**)

* DOI: https://doi.org/10.1145/3814943.3816172
