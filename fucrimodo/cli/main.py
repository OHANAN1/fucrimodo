import argparse
import textwrap
from importlib import import_module

class CLIError(Exception):
    """Error for CLI commands.

    A subcommand may raise this.  The message will be forwarded to
    the error() method of the argument parser."""


commands = [
    ('run', 'fucrimodo.cli.run'),
    ('analyse', 'fucrimodo.cli.analyse')
]


def main(
    prog='fucrimodo', 
    description='Fucrimodo command line tool.', 
    commands=commands,
    args=None
):
    parser = argparse.ArgumentParser(
        prog=prog,description=description,
    )
    subparsers = parser.add_subparsers(
        title='Sub-commands',
        dest='command'
    )
    parser.add_argument('-T', '--traceback', action='store_true')
    subparser = subparsers.add_parser(
        'help',
        description='Help',
        help='Help for sub-command.'
    )
    subparser.add_argument(
        'helpcommand',
        nargs='?',
        metavar='sub-command',
        help='Provide help for sub-command.'
    )

    functions = {}
    parsers = {}
    for command, module_name in commands:
        cmd = import_module(module_name).CLICommand
        docstring = cmd.__doc__
        if docstring is None:
            short = ""
            long = ""
        else:
            parts = docstring.split('\n', 1)
            if len(parts) == 1:
                short = docstring
                long = docstring
            else:
                short, body = parts
                long = short + '\n' + textwrap.dedent(body)

        subparser = subparsers.add_parser(
            command,
            help=short,
            description=long
        )
        cmd.add_arguments(subparser)
        functions[command] = cmd.run
        parsers[command] = subparser

    args = parser.parse_args(args)
    if args.command == 'help':
        if args.helpcommand is None:
            parser.print_help()
        else:
            parsers[args.helpcommand].print_help()
    elif args.command is None:
        parser.print_usage()
    else:
        f = functions[args.command]
        try:
            if f.__code__.co_argcount == 1:
                f(args)
            else:
                f(args, parsers[args.command])
        except KeyboardInterrupt:
            pass
        except CLIError as x:
            parser.error(x)
        except Exception as x:
            if args.traceback:
                raise
            else:
                l1 = f'{x.__class__.__name__}: {x}\n'
                l2 = ('To get a full traceback, use: {} -T {} ...'
                      .format(prog, args.command))
                parser.error(l1 + l2)
