# Debiasing Direct Preference Optimization (DPO)

<img width="1450" height="362" alt="main_figure" src="https://github.com/user-attachments/assets/8e164708-2dad-4689-a3d1-19316c59fc61" />

This repository contains the codebase for Debiasing DPO, introduced in our paper “Mitigating LLM Biases Toward Spurious Social Contexts Using Direct Preference Optimization” ([paper link](https://arxiv.org/abs/2604.02585)).

To run debiasing_dpo, which combines DPO (using the model’s self-generated chosen and rejected reasoning pairs for a spurious context–query pair) and SFT (using the query along with human-provided ground-truth evaluation scores), use the following command:

```
openrlhf.cli.train_dpo \
   --save_path  \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Llama-3.2-3B-Instruct \
   --param_dtype bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset json@ \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --attn_implementation flash_attention_2 \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb 

```
Datasets can be found in this [google drive folder](https://drive.google.com/drive/folders/1daSBCStJ7K3KYXpOgJt107M4fJxNP9gf?usp=sharing).

This codebase is built on top of OpenRLHF. For more details on other algorithm implementations, please visit: https://github.com/OpenRLHF/OpenRLHF. 

```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Zilin Zhu and Xianyu and Weixun Wang and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```
