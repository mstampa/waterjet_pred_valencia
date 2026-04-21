#!/usr/bin/env python3

"""Command-line interface for running fire stream simulations."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import gettempdir
from time import perf_counter

from pandas import DataFrame

from .logging import configure_logging
from .parameters import get_breakup_distance
from .plotting import plot_solution, plot_trace
from .simulator import simulate
from .tracer import Tracer

logger = logging.getLogger("waterjet_pred_valencia.cli")


def main():
    """Entrypoint."""

    configure_logging()
    args: Namespace = get_arguments()

    logger.info("CLI started with the following arguments:")
    for arg in vars(args):
        logger.info(f"    {arg} = {getattr(args, arg)}")

    try:
        run_simulation(args)
    except Exception:
        logger.exception("Simulation run failed.")
        return 1

    logger.info("Exiting.")
    return 0


def run_simulation(args: Namespace) -> None:
    """Run a fire stream simulation with the provided arguments.

    Args:
        argparse.Namespace: Parsed CLI arguments.
    """

    # NOTE: Use bypass to define manual overrides for dyds computations. Example:
    # bypass = {"Uc": -0.01, "theta_s": 0.001}
    bypass: dict[str, float] | None = None

    # Tracer records the state of most variables at regular intervals.
    # The recording is used to generate plots of both successful and failed simulations.
    tracer: Tracer = Tracer()
    s_breakup: float = get_breakup_distance(args.nozzle)

    logger.info("Starting simulation...")
    start_time: float = perf_counter()
    failed_error: Exception | None = None
    result = None
    try:
        result = simulate(
            injection_angle_deg=args.angle,
            injection_speed=args.speed,
            nozzle_diameter=args.nozzle,
            injection_height=args.y0,
            s_span=(0.0, args.span),
            max_step=args.max_step,
            debug=args.debug,
            bypass=bypass,
            tracer=tracer,
        )

    except Exception as e:
        failed_error = e
        logger.error("An error occured and the simulation crashed.")
    finally:
        logger.info(f"Execution time: {perf_counter() - start_time:.6f} sec.")

        # Plot and save.
        trace_df: DataFrame = tracer.to_wide_dataframe()
        if failed_error is None:
            assert result is not None
            assert result.sol is not None
            plot_solution(
                result.sol, result.state_idx, args.output, s_breakup=s_breakup
            )
        elif not trace_df.empty:
            plot_trace(trace_df, args.output, s_breakup=s_breakup)
        else:
            logger.warning("Simulation failed before any trace rows were recorded.")

        if args.csv:
            path_csv: Path = Path(args.csv)
            tracer.to_csv(path_csv)

    # if failed_error is not None:
    #     raise failed_error

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
        help="If provided, trace is exported to this CSV file path.",
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
        "-o",
        "--output",
        help="Plots are saved to this HTML file path. Default: %(default)s.",
        metavar="path",
        type=Path,
        default=Path(gettempdir()) / "simulation_plot.html",
    )
    parser.add_argument(
        "--span",
        help="Simulation span, i.e., maximum value for s [m]. Default: %(default)s.",
        metavar="float",
        type=float,
        default=100.0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    """Entrypoint."""
    raise SystemExit(main())
