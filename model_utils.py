import sys
from models import MultiLayerPerceptron, transformer
import torch.nn.functional as F

def get_model(config, data_dimensions):
    if config.model_type.lower() == 'mlp':
        model = MultiLayerPerceptron(config, data_dimensions)
    elif config.model_type.lower() == 'transformer':
        model = transformer(config, data_dimensions)
    else:
        print('Model ' + config.model_type + ' does not exist!')
        sys.exit()

    return model



def model_epoch(config, dataLoader = None, model=None, optimizer=None, update_gradients = True, iteration_override = 0):

    if update_gradients:
        model.train(True)
    else:
        model.eval()

    err = []

    for i, input in enumerate(dataLoader):
        if config.device.lower() == 'cuda':
            target = input[1].cuda(non_blocking=True)
            input = input[0].cuda(non_blocking=True)
        else:
            target = input[1]
            input = input[0]


        output = model(input.float())[:,0] # reshape output from flat filters to channels * filters per channel
        loss = F.smooth_l1_loss(output.float(), target.float())

        err.append(loss.data)  # record loss

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters


    return err