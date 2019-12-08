# text-to-image-gen-model-acgan
ACGAN based text to image generation model

The implementation of paper:
Conditional Image Synthesis With Auxiliary Classifier GANs
https://arxiv.org/abs/1610.09585

Training commands:
python3 train_acgan.py --output_dir=output --n_epochs=500 --batchSize=100 --cuda --dataset=cifar10 --imageSize=32 --dataroot=datasets --gpu=0

[499/500][499/500] Loss_D: -0.2606 (-0.4218) Loss_G: 3.0220 (0.9407) D(x): 0.5635 D(G(z)): 0.2408 / 0.0567 Acc: 46.0000 (42.3739)

Start GCloud:

gcloud compute --project "curious-context-259106" ssh --zone "us-west1-b" "pytorch-1-vm"

Download file:

gcloud compute scp ooo@pytorch-1-vm:~/text-to-image-gen-model-acgan/output/samples/0000000000008.png .

Sample ImageNet:

python3 train_acgan.py --output_dir=output --n_epochs=500 --batchSize=100 --cuda --dataset=imagenet --imageSize=32 --dataroot=datasets --gpu=0 --sample=noshuffle --netG=output/models/netG_epoch_24.pt


Calculate FID:

python3 fid_score.py -c 0 ../output/samples_gen/image ../output/samples_ori/image

tensorboard --logdir_spec sn:output499_sn_cifar/tensorboard/,b1:output499_b1_cifar/tensorboard/
To see both the logs, I should wait for a while and reopen the windows again




