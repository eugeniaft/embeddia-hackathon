python -m classification_experimental.generic_transformer_classifier \
--dataset ynacc \
--task_name constructiveness \
--dataset_path data/ydata-ynacc-v1_0 \
--pretrained_model EMBEDDIA/crosloengual-bert \
--finetuned_model EMBEDDIA/crosloengual-bert_42_pan_bot \
--max_len 128 \
--prediction \
--use_gpu \
--save_results data/ydata-ynacc-v1_0/bot_results.csv