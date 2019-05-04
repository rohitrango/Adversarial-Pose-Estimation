CUDA_VISIBLE_DEVICES=2 python3 trainModel.py --path /mnt/a99/d0/shriramsb/Documents/academics/sem8/cs763/Adversarial-Pose-Estimation/datasets \
						--modelName first_run \
						--config config.default_config \
						--batch_size 1 \
						--use_gpu \
						--gpu_device 0 \
						--lr 0.0001 \
						--print_every 10 \
						--optimizer_type Adam