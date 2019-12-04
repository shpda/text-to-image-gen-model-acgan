import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torch

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def sample_image(netG, encoder, n_row, batches_done, dataloader, opt):
    """Saves a grid of generated imagenet pictures with captions"""
    target_dir = os.path.join(opt.output_dir, "samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    batch_size = 100

    device = "cpu"
    if opt.cuda:
        device = "cuda"

    captions = []
    gen_imgs = []
    # get sample captions
    done = False
    while not done:
        for (_, labels_batch, captions_batch) in dataloader:

            eval_noise_ = np.random.normal(0, 1, (batch_size, opt.nz))

            captions += captions_batch
            conditional_embeddings = encoder(labels_batch.to(device), captions_batch)

            embeddings = conditional_embeddings.detach().numpy()
            eval_noise_[np.arange(batch_size), :opt.embed_size] = embeddings[:, :opt.embed_size]
            eval_noise_ = (torch.from_numpy(eval_noise_))

            imgs = netG(eval_noise_.view(batch_size, opt.nz, 1, 1).float().to(device)).cpu()
            gen_imgs.append(imgs)

            if len(captions) > n_row ** 2:
                done = True
                break

    gen_imgs = torch.cat(gen_imgs).detach().numpy()
    gen_imgs = np.clip(gen_imgs, 0, 1)

    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.2)

    for i in range(n_row ** 2):
        grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
        grid[i].set_title(captions[i])
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)

    save_file = os.path.join(target_dir, "{:013d}.png".format(batches_done))
    plt.savefig(save_file)
    print("saved  {}".format(save_file))
    plt.close()


