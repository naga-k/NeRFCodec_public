# export CUDA_VISIBLE_DEVICES=7
# train
python train.py --config configs/chair.txt

#python train.py --config configs/chair.txt --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th --render_test 1
# render
#python train.py --config configs/chair.txt --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th --render_only 1 --render_test 1