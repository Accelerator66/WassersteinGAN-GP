from wgan import Generator, Discriminator, weights_init
from config import Cfg
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


if __name__ == "__main__":

    # Read config
    cfg = Cfg()

    # Try to load dataloader from cache
    dataloader_path = os.path.join(cfg.cache, "dataloader.pt")
    if os.path.exists(dataloader_path):
        dataloader = torch.load(dataloader_path)
    else:
        # Create the data set
        dataset = dset.ImageFolder(root=cfg.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(cfg.image_size),
                                       transforms.CenterCrop(cfg.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,
                                                 shuffle=True, num_workers=cfg.workers)
        # Save to cache
        torch.save(dataloader, dataloader_path)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and cfg.ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(cfg.ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (cfg.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(cfg.ngpu)))

    # Initialize all weights to mean=0, std=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(cfg.ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (cfg.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(cfg.ngpu)))

    # Initialize all weights to mean=0, std=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Create batch of latent vectors used to visualize progression of the generator
    fixed_noise = torch.randn(64, cfg.nz, 1, 1, device=device)

    # Setup RMSprop optimizers for both G and D
    optimizerD = optim.RMSprop(netD.parameters(), lr=cfg.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=cfg.lr)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    gen_iterations = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(cfg.num_epochs):
        # For each batch in the data loader
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # Update D network: minimize - D(x) + D(G(z))
            ############################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            j = 0
            while j < cfg.num_dis_iter and i < len(dataloader):
                j += 1
                # Clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(cfg.lower, cfg.upper)
                # Train with all-real batch
                netD.zero_grad()
                data = data_iter.next()
                i += 1
                # Format batch
                real = torch.FloatTensor(data[0])
                real = Variable(real).to(device)
                b_size = real.size(0)
                # Forward pass real batch through D
                output_real = netD(real)
                errD_real = output_real.mean(0).view(1)

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, cfg.nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                # Classify all fake batch with D and disable gradient passing to netG
                output_fake = netD(fake.detach())
                errD_fake = output_fake.mean(0).view(1)
                # Wasserstein distance loss
                errD = - errD_real + errD_fake
                errD.backward()
                optimizerD.step()

            ############################
            # Update G network: minimize -D(G(z))
            ############################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            noise = torch.randn(b_size, cfg.nz, 1, 1, device=device)
            noise = Variable(noise)
            fake = netG(noise)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake)
            errG = - output.mean(0).view(1)
            errG.backward()
            # Update G
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, cfg.num_epochs, i, len(dataloader), gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            if gen_iterations % 500 == 0:
                real = real.mul(0.5).add(0.5)
                vutils.save_image(real, '{0}/real_samples.png'.format(cfg.cache))
                fake = netG(Variable(fixed_noise, volatile=True))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(cfg.cache, gen_iterations))
                torch.save(netD, os.path.join(cfg.model, "D_{0}_{1}.pt".format(gen_iterations, epoch)))
                torch.save(netG, os.path.join(cfg.model, "G_{0}_{1}.pt".format(gen_iterations, epoch)))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Save model per epoch
        torch.save(netD, os.path.join(cfg.model, "D_{0}.pt".format(epoch)))
        torch.save(netG, os.path.join(cfg.model, "G_{0}.pt".format(epoch)))

    # Save final model
    torch.save(netD, os.path.join(cfg.model, "D_final.pt"))
    torch.save(netG, os.path.join(cfg.model, "G_final.pt"))
