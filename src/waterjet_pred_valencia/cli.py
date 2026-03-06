#!/usr/bin/env python3

"""Command-line interface for running fire stream simulations."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import gettempdir
from time import time
from typing import Dict, Optional

import numpy as np

from .plotting import plot_solution
from .simulator import simulate

logger = logging.getLogger("cli")


def main():
    """Entrypoint."""

    args: Namespace = get_arguments()

    # Configure logging.
    log_fmt = "[{asctime}] [{levelname}] [{name}] {message}"
    logging.basicConfig(format=log_fmt, style="{")
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    np.set_printoptions(precision=4)

    # Show input arguments.
    for arg in vars(args):
        logger.info(f"{arg} = {getattr(args, arg)}")

    run_simulation(args)
    return


def run_simulation(args: Namespace) -> None:
    """Run a fire stream simulation with the provided arguments.

    Args:
        argparse.Namespace: Parsed CLI arguments
    """

    bypass: Optional[Dict[str, float]] = None
    if args.debug:
        from .tracer import Tracer

        tracer = Tracer()

        # NOTE: Use bypass to define manual overrides for dyds computations, e.g.:
        # bypass = {"Uc": -0.01, "theta_s": 0.001}
    else:
        tracer = None

    logger.info("Starting simulation...")
    start_time = time()
    try:
        result = simulate(
            args.speed,
            args.angle,
            args.nozzle,
            s_span=(0.0, args.span),
            max_step=args.max_step,
            debug=args.debug,
            bypass=bypass,
            tracer=tracer,
        )

        logger.info(f"Plotting results and saving to {args.output}...")
        assert result.sol is not None
        plot_solution(result.sol, result.state_idx, args.output)

    except Exception as e:
        logger.error(f"An error occured: {e}")
    finally:
        logger.info(f"Execution time: {time() - start_time:.6f} sec.")

        if tracer is not None:
            path_trace = Path(args.trace)
            logger.info(f"Saving trace to {str(path_trace)}")
            tracer.to_csv(path_trace)

    logger.info("All done!")
    return


def get_arguments() -> Namespace:
    """Declare and parse arguments for the CLI.

    Default values are mostly taken from Test 5.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = ArgumentParser(
        description="Simulates a fire stream trajectory and saves the results as\
        interactive plots in an HTML file."
    )

    # Key arguments.
    parser.add_argument(
        "-a",
        "--angle",
        help="theta_0 injection angle above horizon [deg]. Default: %(default)s.",
        metavar="float",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "-n",
        "--nozzle",
        help="D_0 nozzle diameter [m]. Default: %(default)s.",
        metavar="float",
        type=float,
        default=0.0254,
    )
    parser.add_argument(
        "-u",
        "--speed",
        help="U_0 injection speed [m/s]. Default: %(default)s.",
        metavar="float",
        type=float,
        default=30.8,
    )

    # Optional arguments.
    parser.add_argument(
        "-d",
        "--debug",
        help="Activates tracer, console printouts, and auto-drop into PDB on error.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--span",
        help="Simulation span, i.e., maximum value for s [m]. Default: %(default)s.",
        metavar="float",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--max_step",
        help="Max simulation step [m]. Default: %(default)s"
        + " Increase to trade accuracy for performance.",
        metavar="float",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save the generated plots to. Default: %(default)s.",
        metavar="path",
        type=Path,
        default=Path(gettempdir()) / "valencia.html",
    )
    parser.add_argument(
        "--trace",
        help="Path to save the traced variables to. Requires debug mode.\
        Default: %(default)s.",
        metavar="path",
        type=Path,
        default=Path(gettempdir()) / "trace.csv",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """Entrypoint."""
    raise SystemExit(main())
