import wandb
from dataset_utils import build_dataset, build_dataloaders
from model_utils import get_model, model_epoch
import torch.optim as optim
import torch
import time
from utils import check_convergence
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchsummary import summary

class main_container:
    def __init__(self, config):
        self.config = config

    def main(self):
        with wandb.init(project=self.config.project_name, entity="mkilgour", tags = 'sweep test', config = self.config):
            config = wandb.config
            print(config)

            dataset_builder = build_dataset(config)
            train_loader, test_loader = build_dataloaders(dataset_builder, config)
            model = get_model(config, dataset_builder.data_dimensions)
            optimizer = optim.AdamW(model.parameters(), amsgrad=True, lr=config.learning_rate)
            scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = 'min', factor = 0.1, patience = 10, threshold = 1e-3, threshold_mode = 'rel', cooldown = 25, min_lr = 1e-5, verbose=True)

            if config.device.lower() == 'cuda':
                torch.backends.cudnn.benchmark = True
                model.cuda()
                try:
                    print(summary(model,(2,config.dataset_dimension)))
                except:
                    pass

            train_loss_record = []
            test_loss_record = []

            wandb.watch(model, log_graph=True)

            for epoch in range(config.epochs):

                t0 = time.time()
                err_tr = model_epoch(config, dataLoader=train_loader, model=model, optimizer=optimizer, update_gradients=True)  # train & compute loss
                time_train = int(time.time() - t0)

                t0 = time.time()
                err_te = model_epoch(config, dataLoader=test_loader, model=model, update_gradients=False)  # compute loss on test set
                time_test = int(time.time() - t0)

                train_loss_record.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
                test_loss_record.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
                best_test = np.amin(test_loss_record)
                scheduler1.step(best_test)

                wandb.log(
                    {"Train Loss": train_loss_record[-1],
                     "Test Loss": test_loss_record[-1],
                     "Best Test": best_test,
                     "Epoch": epoch}
                )

                if check_convergence(config, test_loss_record) and (epoch > config.history + 2):
                    break

                print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_train, time_test))
