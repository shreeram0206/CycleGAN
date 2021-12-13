import torch
from dataset import Monet2PhotoDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import matplotlib.pyplot as plt


def train_fn(disc_M, disc_R, gen_R, gen_M, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    M_reals = 0
    M_fakes = 0
    loss_array=[]
    loop = tqdm(loader, leave=True)

    for idx, (photo, monet) in enumerate(loop):
        photo = photo.to(config.DEVICE)
        monet = monet.to(config.DEVICE)

        # Train Discriminators Monet and Photo
        with torch.cuda.amp.autocast():
            fake_monet = gen_M(photo)
            D_M_real = disc_M(monet)
            D_M_fake = disc_M(fake_monet.detach())
            M_reals += D_M_real.mean().item()
            M_fakes += D_M_fake.mean().item()
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            fake_photo = gen_R(monet)
            D_P_real = disc_R(photo)
            D_P_fake = disc_R(fake_photo.detach())
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            # Total Discriminator Loss
            D_loss = (D_M_loss + D_P_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators Monet and Photo
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            D_M_fake = disc_M(fake_monet)
            D_P_fake = disc_R(fake_photo)
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))

            # Cycle loss
            cycle_photo = gen_R(fake_monet)
            cycle_monet = gen_M(fake_photo)
            cycle_photo_loss = l1(photo, cycle_photo)
            cycle_monet_loss = l1(monet, cycle_monet)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_photo = gen_R(photo)
            identity_monet = gen_M(monet)
            identity_photo_loss = l1(photo, identity_photo)
            identity_monet_loss = l1(monet, identity_monet)

            # Total Loss
            G_loss = (
                loss_G_P
                + loss_G_M
                + cycle_photo_loss * config.LAMBDA_CYCLE
                + cycle_monet_loss * config.LAMBDA_CYCLE
                + identity_monet_loss * config.LAMBDA_IDENTITY
                + identity_photo_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        save_image(photo*0.5+0.5, f"saved_images/realimage_{idx}.png")
        save_image(fake_monet*0.5+0.5, f"saved_images/generated_monet_{idx}.png")
        # save_image(monet*0.5+0.5, f"saved_images/monetimage_{idx}.png")
        # save_image(fake_photo*0.5+0.5, f"saved_images/generated_real_{idx}.png")
        loop.set_postfix(Monet_real=M_reals/(idx+1), Monet_fake=M_fakes/(idx+1))
    # loss_value = M_fakes / (idx+1)
    # print("Total Loss value is ", G_loss)
    # print(G_loss.item())
    return G_loss, D_loss


def main():
    disc_M = Discriminator(in_channels=3).to(config.DEVICE)
    disc_R = Discriminator(in_channels=3).to(config.DEVICE)
    gen_R = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_M = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_M.parameters()) + list(disc_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_R.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_M, gen_M, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_R, gen_R, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_M, disc_M, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_R, disc_R, opt_disc, config.LEARNING_RATE,
        )

    dataset = Monet2PhotoDataset(
        root_monet=config.TRAIN_DIR+"/monet", root_photo=config.TRAIN_DIR+"/real", transform=config.transforms
    )
    val_dataset = Monet2PhotoDataset(
       root_monet=config.VAL_DIR+"/monet", root_photo=config.VAL_DIR+"/real", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    epochs_array = []
    discriminator_loss_array = []
    G_loss_array = []

    for epoch in range(config.NUM_EPOCHS):
        g_loss, disc_loss = train_fn(disc_M, disc_R, gen_R, gen_M, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        G_loss_array.append(g_loss.cpu().detach().numpy())
        discriminator_loss_array.append(disc_loss.cpu().detach().numpy())
        if config.SAVE_MODEL:
            save_checkpoint(gen_M, opt_gen, filename=config.CHECKPOINT_GEN_M)
            save_checkpoint(gen_R, opt_gen, filename=config.CHECKPOINT_GEN_R)
            save_checkpoint(disc_M, opt_disc, filename=config.CHECKPOINT_CRITIC_M)
            save_checkpoint(disc_R, opt_disc, filename=config.CHECKPOINT_CRITIC_R)

        epochs_array.append(epoch)
        plot1 = plt.figure(1)
        plt.plot(epochs_array, discriminator_loss_array)
        plt.title("Loss vs Number of Epochs (Discriminator)")
        plt.xlabel("Epochs")
        plt.ylabel("Discriminator Loss")
        plt.savefig("Loss_plot_disciminator.png")
        plot2 = plt.figure(2)
        plt.plot(epochs_array, G_loss_array)
        plt.title("Loss vs Number of Epochs (Total Loss)")
        plt.xlabel("Epochs")
        plt.ylabel("Total Loss")
        plt.savefig("Loss_plot_TotalLoss.png")
        plt.show()


if __name__ == "__main__":
    main()
