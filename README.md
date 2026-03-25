# DCE-GTN
The Implementation of "DCE-GTN: A Dynamic Co-evolutionary Graph and Transformer Network for Aspect-based Sentiment Analysis"
DCE-GTN: Dynamic Co-Evolution Graph Transformer Network for Aspect-based Sentiment Analysis

Code for Paper "DCE-GTN: Dynamic Co-Evolution Graph Transformer Network for Aspect-based Sentiment Analysis"

Datasets

Download datasets from these links and put them in the corresponding folder:

- Twitter
- Lap14
- Rest14

Usage

1. Install dependencies

    pip install -r requirements.txt

2. Prepare RoBERTa model
Download RoBERTa base model and put it in ./models/Roberta/
3. The vocabulary files should be in each dataset directory

    python prepare_vocab.py

Training

    # Train on Laptop dataset
    python train.py --dataset laptop --cuda 0
    
    # Train on Restaurant dataset
    python train.py  --dataset restaurant --cuda 0
    
    # Train on Twitter dataset
    python train.py  --dataset twitter --cuda 0


