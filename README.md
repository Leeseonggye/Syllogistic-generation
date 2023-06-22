# Syllogistic generation with Chain-Augmentation using encoder-decoder language model
Project of Text-Analytics, 2022 Spring in DSBA, Industrial &amp; Management Engineering, Korea University 

# Dataset
* Avicenna: a challenge dataset for natural language generation toward commonsense syllogistic reasoning (Zeinab Aghahadi & Alireza Talebpour, 2022) [[Link](https://www.tandfonline.com/doi/full/10.1080/11663081.2022.2041352?src=&journalCode=tncl20)]
* You can get the datasets from [[here](https://github.com/ZeinabAghahadi/Syllogistic-Commonsense-Reasoning)]

# Requirements
We use huggingface transformer docker image. You can simply download [[here](https://hub.docker.com/r/huggingface/transformers-pytorch-gpu)]


# Usage
Step 1. You have to generate augmented dataset through Chain-Augmentation.
```
bash scripts/bart_step1_augement_data_generation.sh
```

Step 2. You have to do k-fold validation to get best epoch.
```
bash scripts/bart_step2_kfold_script_with_augmentation_data.sh
```

Step 3. Train with best epoch.
```
bash scripts/bart_step3_test_with_chain_augmentation_data.sh
```

Step 4. Get ROUGE1/2/L, BLEU and BERTScore through generated conclusion.
```
bash calculate_score_bart.sh
```
