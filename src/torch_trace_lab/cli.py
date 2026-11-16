"""``torch-lab``: command line entry point.

Each subcommand lives in its own module and exposes ``add_parser(subparsers)``,
which registers the subcommand and sets ``func`` to the function that runs it.
"""

import argparse
import sys

from torch_trace_lab import example_traces

SUBCOMMANDS = [example_traces]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torch-lab",
        description="Tools for the SHARCNET PyTorch Profiler and HTA seminar.",
    )
    subparsers = parser.add_subparsers(title="commands", metavar="<command>", required=True)
    for module in SUBCOMMANDS:
        module.add_parser(subparsers)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
