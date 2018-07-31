from argparse import ArgumentParser, Namespace
from configparser import ConfigParser, ExtendedInterpolation
from ParallelMCTS import ParallelMCTS, SinglePlayerEnvironment
from tensorflow import keras
from importlib import import_module
from typing import Tuple, Optional, Callable, Generator
DefinitionType = Optional[Tuple[Callable[[], SinglePlayerEnvironment], Callable[[], keras.Model]]]


def parse_args() -> Tuple[str, str]:
    """ Parse the inputs into MCTSTrain and extract the two input strings.

    Returns
    -------
    definition_file: str
    config_file: str

    """
    parser = ArgumentParser()
    parser.add_argument("definition_file", type=str,
                        help="File containing the environment and network definition.")
    parser.add_argument("config_file", type=str,
                        help="INI file containing MCTS config options.")

    args: Namespace = parser.parse_args()
    definition_file: str = args.definition_file
    config_file: str = args.config_file

    assert definition_file.split('.')[-1] == 'py', "ARGUMENT: definition_file must be a python file."
    assert config_file.split('.')[-1] == 'ini', "ARGUMENT: config_file must be an INI file."

    return definition_file, config_file


def import_definitions(definition_file: str) -> DefinitionType:
    """ Import the two required functions from the definition_file.

    This function looks for make_env() and make_model()

    Parameters
    ----------
    definition_file : str
        Python file containing the required functions.

    Returns
    -------
    make_env : () -> SinglePlayerEnvironment
    make_model: () -> keras.Model

    """
    network_mod_name, _ = definition_file.rsplit(".", 1)

    try:
        network_mod = import_module(network_mod_name)
    except ModuleNotFoundError:
        print("Count not find definitions file: {}".format(definition_file))
        return None

    try:
        make_env = network_mod.make_env
    except AttributeError:
        print("{} did not contain a function named 'make_env'".format(definition_file))
        return None

    try:
        make_model = network_mod.make_model
    except AttributeError:
        print("{} did not contain a function named 'make_model'".format(definition_file))
        return None

    return make_env, make_model


def safe_int(value: str) -> Optional[int]:
    """ Convert a string value into an int but return a None if the string is 'None'.
    """
    return int(value) if value != "None" else None


def safe_float(value: str) -> Optional[float]:
    """ Convert a string value into a float but return a None if the string is 'None'.
        """
    return float(value) if value != "None" else None


# noinspection PyDictCreation
def parse_config(config_file: str) -> dict:
    """ Parse the INI file config and combine the options into an input dictionary.

    Parameters
    ----------
    config_file: str
        File path to the ini file.

    Returns
    -------
    mcts_config: dict
        Dictionary almost ready to be passed into ParallelMCTS.

    """
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_file)

    database_config = {}
    database_config['synchronous'] = config.getboolean('Database', 'synchronous', fallback=True)
    database_config['num_action_threads'] = config.getint('Database', 'num_action_threads', fallback=16)

    network_config = {}
    network_config['learning_rate'] = config.getfloat('Network', 'learning_rate', fallback=0.01)
    network_config['policy_weight'] = config.getfloat('Network', 'policy_weight', fallback=1.0)
    network_config['training_batch_size'] = config.getint('Network', 'training_batch_size', fallback=64)
    network_config['tensorboard_log'] = config.getboolean('Network', 'tensorboard_log', fallback=False)
    network_config['log_dir'] = config.get('Network', 'log_dir', fallback="./logs")

    network_config['checkpoint_steps'] = config.get('Network', 'checkpoint_steps', fallback=None)
    network_config['checkpoint_steps'] = safe_int(network_config['checkpoint_steps'])

    network_config['checkpoint_dir'] = config.get('Network', 'checkpoint_dir', fallback=None)
    network_config['train_buffer_size'] = config.getint('Network', 'train_buffer_size', fallback=64)
    network_config['start_port'] = config.getint('Network', 'start_port', fallback=2222)

    network_config['num_ps'] = config.get('Network', 'num_ps', fallback=None)
    network_config['num_ps'] = safe_float(network_config['num_ps'])

    network_config['batch_size'] = config.get('Network', 'batch_size', fallback=None)
    network_config['batch_size'] = safe_int(network_config['batch_size'])

    mcts_config = {}
    mcts_config['calculation_time'] = config.getfloat('MCTS', 'calculation_time', fallback=15.0)
    mcts_config['C'] = config.getfloat('MCTS', 'C', fallback=1.4)
    mcts_config['temperature'] = config.getfloat('MCTS', 'temperature', fallback=1.0)
    mcts_config['epsilon'] = config.getfloat('MCTS', 'epsilon', fallback=0.3)
    mcts_config['alpha'] = config.getfloat('MCTS', 'alpha', fallback=0.03)
    mcts_config['virtual_loss'] = config.getfloat('MCTS', 'virtual_loss', fallback=2.0)
    mcts_config['verbose'] = config.getint('MCTS', 'verbose', fallback=0)
    mcts_config['single_tree'] = config.getboolean('MCTS', 'single_tree', fallback=False)
    mcts_config['backup_true_value'] = config.getboolean('MCTS', 'backup_true_value', fallback=False)

    global_config = {}
    global_config['num_parallel'] = config.getint('Global', 'num_parallel', fallback=1)
    global_config['num_workers'] = config.getint('Global', 'num_workers', fallback=1)
    global_config['num_networks'] = config.getint('Global', 'num_networks', fallback=1)
    global_config['num_gpu'] = config.getint('Global', 'num_gpu', fallback=0)

    global_config['num_games'] = config.getint('Global', 'num_games', fallback=5)
    global_config['num_epochs'] = config.getint('Global', 'num_epochs', fallback=-1)

    tensorflow_config = ParallelMCTS.GenerateTensorflowConfig(num_networks=global_config['num_networks'],
                                                              num_gpu=global_config['num_gpu'],
                                                              gpu_memory_ratio=1.0,
                                                              growth=True)

    global_config["session_config"] = tensorflow_config
    global_config["network_options"] = network_config
    global_config['database_options'] = database_config
    global_config.update(mcts_config)

    return global_config


def to_infinity() -> Generator:
    """ An extension of the Range iterator that goes to infinity.

    Notes
    -----
    This function probably slows down a lot once you go past 2^63.

    """
    index = 0
    while True:
        yield index
        index += 1


def train(mcts_config: dict) -> None:
    num_games = mcts_config['num_games']
    num_epochs = mcts_config['num_epochs']
    num_parallel = mcts_config['num_parallel']
    num_workers = mcts_config['num_workers']
    num_networks = mcts_config['num_networks']

    mcts = ParallelMCTS(**mcts_config)

    print("=" * 60)
    print("Training Options")
    print("-" * 60)
    print("Parallel Training Loops: {}".format(num_parallel))
    print("Number of Workers: {}".format(num_workers))
    print("Number of Prediction Networks: {}".format(num_networks))
    print()
    print("Number of Epochs: {}".format(num_epochs if num_epochs > 0 else "Infinity"))
    print("Number of Games per epoch: {}".format(num_games * num_parallel))
    print("=" * 60)
    print()
    print(mcts)
    print()
    print(mcts.network_manager)
    print("\n\n")

    loop = to_infinity() if num_epochs < 0 else range(num_epochs)
    with mcts:
        print("\n")
        print("=" * 60)
        print("Starting MCTS")
        print("-" * 60)

        for epoch in loop:
            print("Training Epoch {}".format(epoch))
            mcts.fit_epoch_multi(num_games_per_worker=num_games)


def main():
    definition_file, config_file = parse_args()
    mcts_config = parse_config(config_file)

    definitions = import_definitions(definition_file)
    if definitions is None:
        print("Exiting")
        exit()
    make_env, make_model = definitions

    env = make_env()
    assert isinstance(env, SinglePlayerEnvironment), "make_env must return a subclass of SinglePlayerEnvironment"

    mcts_config['env'] = env
    mcts_config['make_model'] = make_model

    train(mcts_config)


if __name__ == '__main__':
    main()
