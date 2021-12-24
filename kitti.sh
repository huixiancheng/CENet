python train.py -d /yourdataset -ac config/arch/senet-512.yml -n senet-512
python train.py -d /yourdataset -ac config/arch/senet-1024p.yml -n senet-1024 -p logs/senet-512
python train.py -d /yourdataset -ac config/arch/senet-2048p.yml -n senet-2048 -p logs/senet-1024