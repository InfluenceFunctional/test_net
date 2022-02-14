import sys
from models import MultiLayerPerceptron, transformer
import torch.nn.functional as F
import torch

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
        #loss = F.mse_loss(output.float(), target.float())


        err.append(loss.data)  # record loss

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters


    return err



def save_model(model, optimizer):
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/model_ckpt')


def load_model(config, model, optimizer):
    '''
    Check if a checkpoint exists for this model - if so, load it
    :return:
    '''
    checkpoint = torch.load('ckpts/model_ckpt')

    if list(checkpoint['model_state_dict'])[0][0:6] == 'module': # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config.device == 'cuda':
        #model = nn.DataParallel(self.model) # enables multi-GPU training
        print("Using ", torch.cuda.device_count(), " GPUs")
        model.to(torch.device("cuda:0"))
        for state in optimizer.state.values():  # move optimizer to GPU
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    return model, optimizer