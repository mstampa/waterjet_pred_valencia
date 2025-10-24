#!/usr/bin/env python3

"""
Command-line interface for testing, demonstration etc.
"""

from .simulator import simulate
from .plotting import plot_solution

from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from time import time


clilogger = logging.getLogger("cli")


def main():
    log_fmt = "[{asctime}] [{levelname}] [{name}] {message}"
    logging.basicConfig(format=log_fmt, style="{")

    args = get_arguments()
    clilogger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    for arg in vars(args):
        clilogger.info(f"{arg} = {getattr(args, arg)}")

    run_simulation(args)


def run_simulation(args: Namespace) -> None:
    """
    Run a fire stream simulation with the provided arguments.

    Args:
        argparse.Namespace: Parsed CLI arguments
    """

    clilogger.info("Starting simulation...")
    start_time = time()
    sol, idx = simulate(
        args.speed,
        args.angle,
        args.nozzle,
        s_span=(0.0, args.max_s),
        debug=args.debug,
        # NOTE: bypass argument only used for debugging
        # bypass={"Uc": -0.01, "theta_s": 0.001},
    )
    clilogger.info(f"Simulation finished in {time() - start_time:.4f}s.")

    clilogger.info(f"Plotting results in {args.plotpath}...")
    plot_solution(sol, idx, args.plotpath)

    clilogger.info(f"All done!")
    return


def get_arguments() -> Namespace:
    """
    Declares and parses arguments for the CLI.

    Default values are mostly taken from Test 5.

    Returns:
        argparse.Namespace: Parsed CLI arguments
    """

    parser = ArgumentParser(
        description="Simulates a fire stream trajectory and saves the results as\
        interactive plots in an HTML file."
    )

    parser.add_argument(
        "-a",
        "--angle",
        help="theta_0 injection angle above horizon (default: %(default)s) [deg]",
        metavar="float",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="enable debug mode: activates console printouts, auto-drops into PDB on error.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--max_s",
        help="maximum value for s, aka the simulation limit (default: %(default)s) [m]",
        metavar="float",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "-n",
        "--nozzle",
        help="D_0 nozzle diameter (default: %(default)s) [m]",
        metavar="float",
        type=float,
        default=0.0254,
    )
    parser.add_argument(
        "-p",
        "--plotpath",
        help="where to save the plots to (default: %(default)s)",
        metavar="path",
        type=Path,
        default=Path.cwd() / "valencia.html",
    )
    parser.add_argument(
        "-u",
        "--speed",
        help="U_0 injection speed (default: %(default)s) [m/s]",
        metavar="float",
        type=float,
        default=30.8,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
