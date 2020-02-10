import torchvision
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from vae import VAE, weight_init, loss_func

def create_mnist_dataset(dataset_path, bs, is_train):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            dataset_path, train = is_train,
            download = True, transform = torchvision.transforms.ToTensor()
        ),
        batch_size = bs,
        shuffle = True if is_train else False
    )

def train(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = create_mnist_dataset(config.dataset_path, config.batch_size, True)
    eval_dataset = create_mnist_dataset(config.dataset_path, config.batch_size, False)

    net = VAE().to(device)
    full_model_path = config.model_path + config.model_name
    if config.load_pretrained_model:
        net.load_state_dict(torch.load(full_model_path))
        print('Load the pretrained model from %s successfully!' % full_model_path)
    else:
        weight_init(net)
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        print('First time training!')
    net.train()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = config.learning_rate, betas = [0.5, 0.999])

    summary = SummaryWriter(config.summary_path)
    total_iter = 1

    for e in range(1, config.epoch + 1):
        for idx, (x, _) in enumerate(train_dataset):
            x = x.to(device).view(-1, 784)
            optimizer.zero_grad()
            recon_x, mu, logvar = net(x)
            loss = loss_func(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()

            print('[Epoch %d|Train Batch %d] Loss = %.6f' % (e, idx, loss.item()))
            summary.add_scalar('Train/Loss', loss.item(), total_iter)
            total_iter += 1

        if e % 5 == 0:
            net.eval()
            eval_losses = []
            with torch.no_grad():
                for idx, (x, _) in enumerate(eval_dataset):
                    x = x.to(device).view(-1, 784)
                    recon_x, mu, logvar = net(x)
                    loss = loss_func(recon_x, x, mu, logvar)
                    
                    print('[Epoch %d|Eval Batch %d] Loss = %.6f' % (e, idx, loss.item()))
                    eval_losses.append(loss.item())

                mean_eval_loss = np.mean(eval_losses)
                summary.add_scalar('Eval/Loss', mean_eval_loss, e)

            net.train()

        if e % 5 == 0:
            with torch.no_grad():
                fake_z = torch.randn((64, net.nz)).to(device)
                fake_imgs = net.decode(fake_z).view(-1, 1, 28, 28).detach()
                fake_imgs = torchvision.utils.make_grid(fake_imgs, padding = 2, normalize = True).detach().cpu().numpy()
                summary.add_image('Eval/Fake_imgs_after_%d_epochs' % e, fake_imgs, e)

        if e % 2 == 0:
            torch.save(net.state_dict(), full_model_path)

    summary.close()

def fake(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    net = VAE().to(device)
    net.load_state_dict((torch.load(config.model_path + config.model_name)))
    net.eval()

    with torch.no_grad():
        fake_z = torch.randn((64, net.nz)).to(device)
        fake_imgs = net.decode(fake_z).view(-1, 1, 28, 28).detach()
        plt.figure(figsize = (8, 8))
        plt.axis('off')
        plt.title('Fake images')
        plt.imshow(np.transpose(torchvision.utils.make_grid(fake_imgs, padding = 2, normalize = True).cpu(), (1, 2, 0)))
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type = bool, default = True)
    parser.add_argument('--load_pretrained_model', type = bool, default = False)
    parser.add_argument('--model_path', type = str, default = './model/')
    parser.add_argument('--model_name', type = str, default = 'model.pkl')
    parser.add_argument('--summary_path', type = str, default = './summary/')
    parser.add_argument('--dataset_path', type = str, default = './data/')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--epoch', type = int, default = 30)
    parser.add_argument('--learning_rate', type = float, default = 1e-2)

    config = parser.parse_args()
    if config.is_train:
        train(config)
    else:
        fake(config)