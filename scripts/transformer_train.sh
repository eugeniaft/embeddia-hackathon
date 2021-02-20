python -m classification_experimental.generic_transformer_classifier \
--dataset cro_train_large \
--task_name noblock_large_crotian_fine_tuned_en \
--random_seed 42 \
--pretrained_model /home/isspek/Downloads/crosloengual-bert_42_toxicity_allENdata_2e-05_128 \
--max_len 128 \
--num_label 2 \
--epochs 3 \
--lr 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--weight_decay 0.01 \
--fine_tune \
--tokenizer EMBEDDIA/crosloengual-bert \
--nosplit

python -m classification_experimental.generic_transformer_classifier \
--dataset cro_train_large \
--task_name noblock_large_crotian_embeddia \
--random_seed 42 \
--pretrained_model EMBEDDIA/crosloengual-bert \
--max_len 128 \
--num_label 2 \
--epochs 3 \
--lr 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--weight_decay 0.01 \
--fine_tune \
--tokenizer EMBEDDIA/crosloengual-bert \
--nosplit

python -m classification_experimental.generic_transformer_classifier \
--dataset cro_train \
--task_name noblock_small_crotian_fine_tuned_en \
--random_seed 42 \
--pretrained_model /home/isspek/Downloads/crosloengual-bert_42_toxicity_allENdata_2e-05_128 \
--max_len 128 \
--num_label 2 \
--epochs 3 \
--lr 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--weight_decay 0.01 \
--fine_tune \
--tokenizer EMBEDDIA/crosloengual-bert \
--nosplit

python -m classification_experimental.generic_transformer_classifier \
--dataset est_train_large \
--task_name noblock_large_est_fine_tuned_en \
--random_seed 42 \
--pretrained_model /home/isspek/Downloads/finest-bert_42_toxicity_allENdata_2e-05_128 \
--max_len 128 \
--num_label 2 \
--epochs 3 \
--lr 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--weight_decay 0.01 \
--fine_tune \
--tokenizer EMBEDDIA/finest-bert \
--nosplit

python -m classification_experimental.generic_transformer_classifier \
--dataset est_train \
--task_name noblock_small_est_fine_tuned_en \
--random_seed 42 \
--pretrained_model /home/isspek/Downloads/finest-bert_42_toxicity_allENdata_2e-05_128 \
--max_len 128 \
--num_label 2 \
--epochs 3 \
--lr 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--weight_decay 0.01 \
--fine_tune \
--tokenizer EMBEDDIA/finest-bert \
--nosplit

python -m classification_experimental.generic_transformer_classifier \
--dataset est_train \
--task_name noblock_small_est_embeddia \
--random_seed 42 \
--pretrained_model EMBEDDIA/finest-bert \
--max_len 128 \
--num_label 2 \
--epochs 3 \
--lr 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--weight_decay 0.01 \
--fine_tune \
--tokenizer EMBEDDIA/finest-bert \
--nosplit

python -m classification_experimental.generic_transformer_classifier \
--dataset est_train_large \
--task_name noblock_large_est_embeddia \
--random_seed 42 \
--pretrained_model EMBEDDIA/finest-bert \
--max_len 128 \
--num_label 2 \
--epochs 3 \
--lr 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--weight_decay 0.01 \
--fine_tune \
--tokenizer EMBEDDIA/finest-bert \
--nosplit

#python -m classification_experimental.generic_transformer_classifier \
#--dataset en_toxic \
#--task_name toxicity \
#--random_seed 42 \
#--pretrained_model EMBEDDIA/crosloengual-bert \
#--max_len 128 \
#--num_label 2 \
#--epochs 3 \
#--lr 2e-5 \
#--per_device_train_batch_size 16 \
#--per_device_eval_batch_size 16 \
#--weight_decay 0.01 \
#--fine_tune \
#--nosplit

#python -m classification_experimental.generic_transformer_classifier \
#--dataset pan_bot \
#--task_name pan_bot \
#--dataset_path data/pan_bot \
#--random_seed 42 \
#--pretrained_model EMBEDDIA/crosloengual-bert \
#--max_len 128 \
#--num_label 2 \
#--epochs 3 \
#--lr 2e-5 \
#--per_device_train_batch_size 16 \
#--per_device_eval_batch_size 16 \
#--weight_decay 0.01 \
#--fine_tune
