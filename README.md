How can we use transformers for sequence learning tasks such as sentence classification
 
 
1. To run the fine-tuning of roberta-base on wikitext run: python run_mlm.py --model_name_or_path roberta-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir /tmp/test-mlm 