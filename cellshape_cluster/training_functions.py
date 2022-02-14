import torch
from tqdm import tqdm

from .helpers.distributions import get_distributions, get_target_distribution
from .helpers.kmeans import kmeans


def train(model,
          dataloader,
          dataloader_inf,
          num_epochs,
          optimizer,
          reconstruction_criterion,
          cluster_criterion,
          update_interval,
          gamma,
          q_pow,
          save_to):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = 1000000000

    for epoch in range(num_epochs):
        if epoch == 0:
            print('Performing k-means to get initial cluster centres')
            km = kmeans(model, dataloader)

        if epoch % update_interval == 0:
            print('Updating target distribution')
            cluster_distribution, _ = get_distributions(model, dataloader_inf)
            target_distribution = get_target_distribution(cluster_distribution, q_pow)
        print(f'Training epoch {epoch}')
        batch_num = 1
        running_loss = 0.0

        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]
                tar_dist = torch.from_numpy(
                    target_distribution[((batch_num - 1)
                                         * batch_size):(batch_num * batch_size), :]).to(device)

                # ===================forward=====================
                with torch.set_grad_enabled(True):
                    output, features, clusters = model(inputs)
                    optimizer.zero_grad()
                    reconstruction_loss = reconstruction_criterion(inputs, output)
                    cluster_loss = cluster_criterion(torch.log(clusters), tar_dist)
                    loss = ((1 - gamma) * reconstruction_loss) \
                           + (gamma * cluster_loss)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                running_loss += loss.detach().item() / batch_size
                batch_num += 1
                tepoch.set_postfix(loss=loss.detach().item() / batch_size,
                                   reconstruction_loss=reconstruction_loss.item() / batch_size,
                                   cluster_loss=cluster_loss.item() / batch_size)

            total_loss = running_loss / len(dataloader)
            if total_loss < best_loss:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": total_loss,
                }
                best_loss = total_loss
                torch.save(checkpoint, save_to + ".pt")
