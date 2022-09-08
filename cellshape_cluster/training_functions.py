import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

from distributions import get_distributions, get_target_distribution
from kmeans import kmeans
from check_tolerance import check_tolerance


def train(
    model,
    dataloader,
    dataloader_inf,
    num_epochs,
    optimizer,
    reconstruction_criterion,
    cluster_criterion,
    update_interval,
    gamma,
    divergence_tolerance,
    logging_info,
):

    name_logging, name_model, name_writer, name = logging_info
    writer = SummaryWriter(log_dir=name_writer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = float("inf")
    niter = 1
    # initialise cluster centres with k-means
    logging.info("Performing k-means to get initial cluster centres")
    print("Performing k-means to get initial cluster centres")
    _ = kmeans(model, dataloader)

    # initialise target distribution
    logging.info("Initialising target distribution")
    print("Initialising target distribution")
    cluster_distribution, previous_cluster_predictions = get_distributions(
        model, dataloader_inf
    )
    target_distribution = get_target_distribution(cluster_distribution)

    for epoch in range(num_epochs):

        if (epoch % update_interval == 0) and (epoch != 0):
            logging.info("Updating target distribution")
            cluster_distribution, cluster_predictions = get_distributions(
                model, dataloader_inf
            )
            target_distribution = get_target_distribution(cluster_distribution)
            delta_label, previous_cluster_predictions = check_tolerance(
                cluster_predictions, previous_cluster_predictions
            )
            logging.info(f"Delta label == {delta_label}")
            if delta_label < divergence_tolerance:
                logging.info(
                    f"Label divergence {delta_label} < "
                    f"divergence tolerance {divergence_tolerance}"
                )
                print("Reached tolerance threshold. Stopping training.")
                logging.info(
                    f"Label divergence {delta_label} < "
                    f"divergence tolerance {divergence_tolerance}"
                    f"Reached tolerance threshold. Stopping training."
                )
                break

        print(f"Training epoch {epoch}")
        logging.info(f"Training epoch {epoch}")
        batch_num = 1
        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_cluster = 0.0

        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[0]
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]
                tar_dist = torch.from_numpy(
                    target_distribution[
                        ((batch_num - 1) * batch_size) : (
                            batch_num * batch_size
                        ),
                        :,
                    ]
                ).to(device)

                # ===================forward=====================
                with torch.set_grad_enabled(True):
                    output, features, clusters = model(inputs)
                    optimizer.zero_grad()
                    reconstruction_loss = reconstruction_criterion(
                        inputs, output
                    )
                    cluster_loss = cluster_criterion(
                        torch.log(clusters), tar_dist
                    )
                    loss = reconstruction_loss + (gamma * cluster_loss)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                batch_loss = loss.detach().item() / batch_size
                batch_loss_rec = (
                    reconstruction_loss.detach().item() / batch_size
                )
                batch_loss_cluster = cluster_loss.detach().item() / batch_size

                running_loss += batch_loss
                running_loss_rec += batch_loss_rec
                running_loss_cluster += batch_loss_cluster
                batch_num += 1
                tepoch.set_postfix(
                    tot_loss=loss.detach().item() / batch_size,
                    rec_loss=reconstruction_loss.item() / batch_size,
                    clu_loss=cluster_loss.item() / batch_size,
                )
                writer.add_scalar("/Loss", batch_loss, niter)
                niter += 1

                if batch_num % 10 == 0:
                    logging.info(
                        f"[{epoch}/{num_epochs}]"
                        f"[{batch_num}/{len(dataloader)}]"
                        f"LossTot: {batch_loss}"
                        f"LossRec: {batch_loss_rec}"
                        f"LossCluster: {batch_loss_cluster}"
                    )

            total_loss = running_loss / len(dataloader)
            if total_loss < best_loss:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": total_loss,
                }
                best_loss = total_loss
                torch.save(checkpoint, name_model + ".pt")
                logging.info(
                    f"Saving model to {name_model} with loss = {best_loss}."
                )
