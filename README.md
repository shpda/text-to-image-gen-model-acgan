# text-to-image-gen-model-acgan
ACGAN based text to image generation model

The implementation of paper:
Conditional Image Synthesis With Auxiliary Classifier GANs
https://arxiv.org/abs/1610.09585

Training commands:
python3 main.py --outf=output --niter=500 --batchSize=100 --cuda --dataset=cifar10 --imageSize=32 --dataroot=datasets --gpu=0
