#!/usr/bin/env python3

"""Command-line interface for the interactive fire stream plotting app."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .logging import configure_logging
from .plotting.session import start_server

logger = logging.getLogger("waterjet_pred_valencia.cli")


def main():
    """Start interactive plotting app and return process-style exit code."""

    configure_logging()
    args: Namespace = get_arguments()

    logger.info("CLI started with the following arguments:")
    for arg in vars(args):
        logger.info(f"    {arg} = {getattr(args, arg)}")

    try:
        start_server(
            injection_angle_deg=args.angle,
            injection_speed=args.speed,
            nozzle_diameter=args.nozzle,
            injection_height=args.y0,
            span=args.span,
            max_step=args.max_step,
            debug=args.debug,
            csv_path=args.csv,
            port=args.port,
            show=args.show,
        )
    except Exception:
        logger.exception("Interactive plotting app failed.")
        return 1

    logger.info("Exiting.")
    return 0


def get_arguments() -> Namespace:
    """Declare and parse arguments for the CLI.

    Default values are mostly taken from Test 5. The CLI now starts a local
    Panel app instead of exporting a static HTML plot.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = ArgumentParser(
        description="Starts a local interactive simulation app for the Valencia water jet model."
    )

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
    parser.add_argument(
        "--y0",
        help="Initial height y_0 [m]. Default: %(default)s.",
        metavar="float",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--csv",
        help="If provided, trace is exported to this CSV file path after each run.",
        required=False,
        metavar="trace.csv",
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="If provided, activates live console printouts "
        + "and auto-dropping into PDB on error.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--max_step",
        help="Maximal integration step [m]. Default: %(default)s."
        + " Increase to trade accuracy for performance.",
        metavar="float",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--port",
        help="Port for the local Panel server. Default: %(default)s.",
        metavar="int",
        type=int,
        default=5007,
    )
    parser.add_argument(
        "--span",
        help="Simulation span, i.e., maximum value for s [m]. Default: %(default)s.",
        metavar="float",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--no-show",
        help="If provided, do not auto-open the browser.",
        action="store_false",
        dest="show",
        default=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    """Entrypoint."""
    raise SystemExit(main())
