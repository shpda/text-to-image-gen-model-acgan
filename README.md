# text-to-image-gen-model-acgan
ACGAN based text to image generation model

The implementation of paper:
Conditional Image Synthesis With Auxiliary Classifier GANs
https://arxiv.org/abs/1610.09585

Training commands:
python3 main.py --outf=output --niter=500 --batchSize=100 --cuda --dataset=cifar10 --imageSize=32 --dataroot=datasets --gpu=0

[499/500][499/500] Loss_D: -0.2606 (-0.4218) Loss_G: 3.0220 (0.9407) D(x): 0.5635 D(G(z)): 0.2408 / 0.0567 Acc: 46.0000 (42.3739)

