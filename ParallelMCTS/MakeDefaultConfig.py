from argparse import ArgumentParser
from ParallelMCTS import ParallelMCTS


def dict_to_ini(config: dict, out_file: str) -> None:
    with open(out_file, 'w') as config_file:
        for section_key, section in config.items():
            print("[{}]".format(section_key), file=config_file)
            for config_key, config_default in section.items():
                print("{} = {}".format(config_key, config_default), file=config_file)
            print("\n", file=config_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("output_file", type=str)
    output_file: str = parser.parse_args().output_file

    database_options = ParallelMCTS.DatabaseOptions()
    network_options = ParallelMCTS.NetworkOptions()
    mcts_options = ParallelMCTS.MCTSOptions()
    global_options = {
        "num_parallel": 1,
        "num_workers": 1,
        "num_networks": 1,
        "num_gpu": 0
    }

    options = {
        "Database": database_options,
        "Network": network_options,
        "MCTS": mcts_options,
        "Global": global_options
    }

    print("Writing File")
    dict_to_ini(options, output_file)

    print("Done.")
