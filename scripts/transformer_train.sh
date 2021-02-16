python -m classification_experimental.generic_transformer_classifier \
--dataset en_toxic \
--task_name toxicity \
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
--nosplit

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
