#python -m classification_experimental.generic_transformer_classifier \
#--dataset cro_train_large \
#--pretrained_model EMBEDDIA/crosloengual-bert \
#--finetuned_model /home/isspek/Downloads/crosloengual-bert_42_toxicity_allENdata_2e-05_128_42_noblock_large_crotian_fine_tuned_en_2e-05_128 \
#--max_len 128 \
#--prediction \
#--use_gpu \
#--random_seed 42 \
#--save_results /home/isspek/Downloads/crosloengual-bert_42_toxicity_allENdata_2e-05_128_42_noblock_large_crotian_fine_tuned_en_2e-05_128

#python -m classification_experimental.generic_transformer_classifier \
#--dataset cro_train \
#--pretrained_model EMBEDDIA/crosloengual-bert \
#--finetuned_model /home/isspek/Downloads/crosloengual-bert_42_toxicity_allENdata_2e-05_128_42_noblock_small_crotian_fine_tuned_en_2e-05_128 \
#--max_len 128 \
#--prediction \
#--use_gpu \
#--random_seed 42 \
#--save_results /home/isspek/Downloads/crosloengual-bert_42_toxicity_allENdata_2e-05_128_42_noblock_small_crotian_fine_tuned_en_2e-05_128

python -m classification_experimental.generic_transformer_classifier \
--dataset cro_train_large \
--pretrained_model EMBEDDIA/crosloengual-bert \
--finetuned_model EMBEDDIA/crosloengual-bert_42_noblock_large_crotian_embeddia_2e-05_128 \
--max_len 128 \
--prediction \
--use_gpu \
--random_seed 42 \
--save_results EMBEDDIA/crosloengual-bert_42_noblock_large_crotian_embeddia_2e-05_128

python -m classification_experimental.generic_transformer_classifier \
--dataset cro_train \
--pretrained_model EMBEDDIA/crosloengual-bert \
--finetuned_model EMBEDDIA/crosloengual-bert_42_noblock_2e-05_128 \
--max_len 128 \
--prediction \
--use_gpu \
--random_seed 42 \
--save_results EMBEDDIA/crosloengual-bert_42_noblock_2e-05_128