import copy
import itertools
from functools import reduce
from importlib import import_module
from logging import Logger
from typing import Any, List

import torch
import yaml


def create_class_instance(module_name, class_name, kwargs, *args):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param class_name:
    :param kwargs: arguments needed for the class constructor
    :returns: instance of 'class_name'

    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance


def create_instance(name, params, *args):
    """Creates an instance of class given configuration.

    :param name: of the module we want to create
    :param params: dictionary containing information how to instantiate the class
    :returns: instance of a class
    :rtype:

    """
    i_params = params[name]
    if type(i_params) is list:
        instance = [create_class_instance(
                p['module'], p['name'], p['args'], *args) for p in i_params]
    else:
        instance = create_class_instance(
                i_params['module'], i_params['name'], i_params['args'], *args)
    return instance


def load_params(path: str, logger: Logger) -> dict:
    """Loads experiment parameters from json file.

    :param path: to the json file
    :param logger:
    :returns: param needed for the experiment
    :rtype: dictionary

    """
    try:
        with open(path, "rb") as f:
            params = yaml.full_load(f)
        return params
    except Exception as e:
        logger.error(e)


def get_device(params: dict, rank: int = 0, logger: Logger = None, no_cuda: bool = False) -> torch.device:
    """

    :param params:
    :param logger:
    :return: returns the device
    """
    gpus = params.get("gpus", [])
    if len(gpus) > 0:
        if not torch.cuda.is_available():
            if not str(gpus[rank])=="mps" or no_cuda:
                if logger is not None:
                    logger.warning("No GPU's available. Using CPU.")
                device = torch.device("cpu")
            else:
                device=torch.device("mps")
        else:
            device = torch.device("cuda:" + str(gpus[rank]))
    else:
        device = torch.device("cpu")
    return device


def is_primitive(v: Any) -> bool:
    """
    Checks if v is of primitive type.
    """
    return isinstance(v, (int, float, bool, str))


def free_params(module: Any) -> None:
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = True


def frozen_params(module: Any) -> None:
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = False


def sum_dictionaries(dicts: List) -> dict:
    """
    Sums the values of the common keys in dictionary.
    :param dicts: dictionaries containing numeric values
    :return: dictionary with summed values
    """

    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    return reduce(reducer, dicts, {})


def unpack_cv_parameters(params, prefix=None):
    cv_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = unpack_cv_parameters(value, prefix)
            if '.' in prefix:
                prefix = prefix.rsplit('.', 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                cv_params.extend(param_pool)
        elif isinstance(value, tuple) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = unpack_cv_parameters(v, prefix)
                    if '.' in prefix:
                        prefix = prefix.rsplit('.', 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        cv_params.extend(param_pool)
        elif isinstance(value, list):
            if prefix is None:
                prefix = key
            else:
                key = ".".join([prefix, key])
            cv_params.append([(key, v) for v in value])
    return cv_params


def dict_set_nested(d, keys, value):
    node = d
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return d
        else:
            if "#" in key:
                key, _id = key.split("#")
                if key not in node:
                    node[key] = dict()
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if key not in node:
                    node[key] = dict()
                    node = node[key]
                else:
                    node = node[key]


def convert_tuples_2_list(arg):
    for key, value in arg.items():
        if isinstance(value, dict):
            convert_tuples_2_list(value)
        else:
            if isinstance(value, tuple):
                arg[key] = list(value)

    return arg


def shuffle_tensor(x: torch.tensor, dim: int = 0) -> torch.tensor:
    return x[torch.randperm(x.shape[dim])]


def expand_params(params):
    """
    Expand the hyperparamers for grid search

    :param params:
    :return:
    """
    cv_params = []
    param_pool = unpack_cv_parameters(params)

    for i in list(itertools.product(*param_pool)):
        d = copy.deepcopy(params)
        name = d['name']
        for j in i:
            dict_set_nested(d, j[0].split("."), j[1])
            name += "_" + j[0] + "_" + str(j[1])
            d['name'] = name.replace('.args.', "_")
        d = convert_tuples_2_list(d)
        cv_params.append(d)
    if not cv_params:
        return [params] * params['num_runs']

    gs_params = []
    for p in cv_params:
        gs_params += [p] * p['num_runs']
    return gs_params


def gumbel_softmax(pi, tau, device):
    """
    Gumbel-Softmax distribution.
    Implementation from https://github.com/ericjang/gumbel-softmax.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
    """
    y = gumbel_softmax_sample(pi, tau, device)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return y, y_hard - y.detach() + y


def gumbel_softmax_sample(pi, tau, device, epsilon=1e-12):
    """
    Sample Gumbel-softmax
    """
    y = torch.log(pi + epsilon) + gumbel_sample(pi.size(), device)
    return torch.nn.functional.softmax(y / tau, dim=-1)


def gumbel_sample(shape, device, epsilon=1e-20):
    """
    Sample Gumbel(0,1)
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + epsilon) + epsilon)


def gumbel_soft_perm(pi, tau, device):
    """
    Differentiable permutation sampling based on repeated sampling without replacement from pi.
    Based on Gumbel-Softmax and Top-K sampling. Idea from https://arxiv.org/abs/2203.08509.
    returns: permutation matrix indicating samples without replacement from pi
    [..., s, s], [..., s, s] where s = pi.shape[-1]
    """
    # Gumbel perturbation
    y = gumbel_softmax_sample(pi, tau, device)

    shape_y = y.shape
    y = y.flatten(end_dim=-2)

    # Soft-Sort (https://arxiv.org/abs/2006.16038)
    y_sorted, _ = torch.sort(y, descending=True, stable=True)

    p = - torch.abs(y.unsqueeze(-2) - y_sorted.unsqueeze(-1))
    p_soft = torch.nn.functional.softmax(p / tau, dim = -1)

    # one-hots
    p_hard = torch.zeros_like(p_soft)
    p_hard.scatter_(-1, p_soft.topk(1,-1)[1], 1)
    p_hard = (p_hard - p_soft).detach() + p_soft

    # Reshape for permutation matrix
    p_hard = p_hard.view(*shape_y, shape_y[-1])
    p_soft = p_soft.view(*shape_y, shape_y[-1])

    return p_soft, p_hard

def create_nonlinearity(name):
    """
    Returns instance of non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz()

    return instance


def get_class_nonlinearity(name):
    """
    Returns non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)

    return clazz


def count_lines_in_file(file_path):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))


def clip_grad_norm(parameters, optimizer: dict) -> None:
    if optimizer['grad_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(parameters, optimizer['grad_norm'])


class OneHotEncoding(object):
    def __init__(self, encoding_size: int, ignore_index: int = -1):
        self.encoding_size = encoding_size
        self.ignore_index = ignore_index

    def __call__(self, indexes: torch.LongTensor) -> torch.FloatTensor:
        one_hot = torch.nn.functional.one_hot(indexes, self.encoding_size + 1)

        return one_hot[:, :, :-1].float()

    @property
    def embedding_dim(self):
        return self.encoding_size


def chunk_docs(n_docs: int, chunk_size: int):
    for i in range(0, n_docs, chunk_size):
        yield [i, min(i + chunk_size, n_docs)]
