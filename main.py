from mainfile import main_container
from argparse import ArgumentParser
from utils import add_arg_list, load_yaml
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# parse config
def get_args(parser):
    arg_list = [
        {'name': 'config_file', 'type': str, 'default': 'default_config.yaml'},

        {'name': 'data_source', 'type': str, 'default': 'gaussian_1'}, # 'synthetic', 'gaussian_1'
        {'name': 'run_num', 'type': int, 'default': 0},
        {'name': 'dataset_seed', 'type': int, 'default': 0},
        {'name': 'model_seed', 'type': int, 'default': 0},
        {'name': 'device', 'type': str, 'default': 'cpu'}, # 'cuda' 'cpu'

        # wandb logging / sweeping info
        {'name': 'wandb', 'type': bool, 'default': True}, # log to wandb
        {'name': 'sweep', 'type': bool, 'default': False},
        {'name': 'sweep_config_file', 'type': str, 'default': 'sweep_config.yaml'},
        {'name': 'sweep_id', 'type': str, 'default': None},
        {'name': 'sweep_num_runs', 'type': int, 'default': 10},
        {'name': 'experiment_tag', 'type': str, 'default': 'dev_test_2'},
        {'name': 'project_name', 'type': str, 'default': 'my-test-net'},

        # dataset information for synthetic data generation
        {'name': 'dataset_size', 'type': int, 'default': 10000},
        {'name': 'dataset_dimension', 'type': int, 'default': 5},

        # training details
        {'name': 'batch_size', 'type': int, 'default': 10},
        {'name': 'epochs', 'type': int, 'default': 10000},
        {'name': 'history', 'type': int, 'default': 100},
        {'name': 'learning_rate', 'type': float, 'default': 1e-3},

        # model details
        {'name': 'model_type', 'type': str, 'default': 'mlp'},
        {'name': 'model_layers', 'type': int, 'default': 2},
        {'name': 'model_filters', 'type': int, 'default': 64},
        {'name': 'model_norm', 'type': str, 'default': None},
        {'name': 'model_dropout', 'type': float, 'default': 0},
        {'name': 'model_activation', 'type': str, 'default': 'kernel'}, # 'relu' 'gelu' 'kernel'

    ]

    parser = add_arg_list(parser,arg_list)

    return parser

# run main
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = get_args(parser)

    config = parser.parse_args()
    if config.config_file is not None:  # load up config from file
        yaml_config = load_yaml(config.config_file)
        for key in yaml_config.keys():  # overwrite config with values from yaml
            vars(config)[key] = yaml_config[key]

    container = main_container(config)

    if not config.sweep:
        container.main()
    else:
        sweep_config = load_yaml(config.sweep_config_file)
        wandb.login()
        if config.sweep_id is not None:
            sweep_id = config.sweep_id
        else:
            sweep_id = wandb.sweep(sweep_config, project=config.project_name)
        for sweep_run in range(config.sweep_num_runs):
            wandb.agent(sweep_id, container.main, count=1, project = config.project_name)