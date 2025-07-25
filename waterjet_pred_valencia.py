#!/usr/bin/env python3

"""
User interface for testing, demonstration etc.
"""

from model.simulator import simulate
from plotting import plot_solution

from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import time


def main(args: Namespace) -> None:
    """
    Run a fire stream simulation with default parameters (Test 5).

    Parameters:
    - args: arguments from argparse
    """

    start_time = time()
    sol, idx = simulate(
        args.speed,
        args.angle,
        args.nozzle,
        s_span=(0.0, args.max_s),
        debug=args.debug,
    )
    print(f"\nSimulation finished after {time() - start_time:.4f}s.")

    plot_solution(sol, idx, args.plotpath)

    print(f"All done!")
    return


def setup_arguments() -> ArgumentParser:
    """
    Prepares arguments for the CLI.

    Returns:
    - ArgumentParser
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
        help="enable debug mode (activates console printouts, auto-drops into PDB).",
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

    return parser


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = setup_arguments()
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))
    print("")

    main(args)
