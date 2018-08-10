from argparse import ArgumentParser, Namespace
from configparser import ConfigParser, ExtendedInterpolation
from ast import literal_eval


def get_data_type(value):
    value = value.strip()

    if len(value) == 0:
        raise ValueError("Empty Value")

    try:
        t = literal_eval(value)
    except ValueError:
        return "get"
    except SyntaxError:
        return "get"

    else:
        if type(t) in [int, float, bool]:
            if type(t) is int:
                return "getint"
            if type(t) is float:
                return "getfloat"
            if t in {True, False}:
                return "getboolean"
        else:
            return "get"

def generate_parse_function(config: ConfigParser):
    for section in config.sections():
        var_name = "{}_config".format(section.lower())
        print("{} = {}".format(var_name, "{}"))
        for config_name, config_default in config[section].items():
            print("{var_name}['{name}'] = config.{function}('{section}', '{name}', fallback={default})".format(
                var_name=var_name,
                name=config_name,
                function=get_data_type(config_default),
                section=section,
                default=config_default
            ))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("default_ini", type=str)

    default_ini: str = parser.parse_args().default_ini
    config = ConfigParser()
    config.read(default_ini)

    generate_parse_function(config)
