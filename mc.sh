
export SWAG_DIR=../MR/RACE/
python run_multiple_choice.py \
	--task_name race \
	--model_name_or_path roberta-base \
	--do_train \
	--do_eval \
	--data_dir $SWAG_DIR \
	--learning_rate 5e-5 \
	--num_train_epochs 3 \
	--max_seq_length 1500 \
	--output_dir models_bert/race_base \
	--per_gpu_eval_batch_size=16 \
	--per_device_train_batch_size=16 \
	--gradient_accumulation_steps 2 \
	--overwrite_output

