import numpy as np
from pathlib import Path
from argparse import Namespace
import yaml

def check_convergence(config, record):
    converged = False

    if len(record) > config.history:
        if all(record[-config.history:] > np.amin(record)):
            converged = True
            print("Model converged, test loss increasing")

        if np.var(record[-config.history:]) / np.average(record[-config.history:]) < 1e-5:
            converged = True
            print("Model converged, test loss stabilized")

    return converged

def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action = 'store_true')
    group.add_argument('--no-' + name, dest=name, action = 'store_false')
    parser.set_defaults(**{name:default})

def add_arg_list(parser, arg_list):
    for entry in arg_list:
        if entry['type'] == 'bool':
            add_bool_arg(parser, entry['name'], entry['default'])
        else:
            parser.add_argument('--' + entry['name'], type = entry['type'], default = entry['default'])

    return parser


def dict2namespace(data_dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace

def load_yaml(path):
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict

def standardize(y):
    return (y - np.average(y)) / np.sqrt(np.var(y))