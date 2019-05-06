CUDA_VISIBLE_DEVICES=2 
python3 trainModel.py \
--path lspet_dataset \
--modelName first_run \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--gpu_device 0 \
--lr 0.0001 \
--print_every 10 \
--optimizer_type Adam