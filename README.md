# Fire stream trajectory model — Valencia et al

**NOTE:**
Work in progress and currently not functional!
There seem to be errors in the ODE system.
Example plots were generated with some of the computations bypassed.

This is a Python module for simulating the trajectories of "fire streams",
i.e., large water jets shot from fire monitors.
The backbone is a 1D Eulerian analytical model reproduced from this publication:

```bibtex
@article{valencia_model_22,
  title = {A Model for Predicting the Trajectory and Structure of Firefighting Hose Streams},
  author = {Valencia, Andres and Zheng, Yinghui and Marshall, André W.},
  date = {2022-03-01},
  journaltitle = {Fire Technology},
  volume = {58},
  number = {2},
  pages = {793--815},
  issn = {1572-8099},
  doi = {10.1007/s10694-021-01175-1},
}
```

The model describes the fire stream as a combination of three phases:
water-core phase, air phase, and spray phase.
It includes air entrainment, jet break-up spray generation
and multi-dispersion (i.e., droplets of multiple sizes).
Trajectory and spray behavior are modeled over the streamwise axis "s".

![Example plots](doc/example_plots.png)

## Getting started

### Prerequisites

Key dependencies of this package are:

* [SciPy](https://scipy.org) for solving the ODE
* [bokeh](https://bokeh.org) for plotting

Developers wishing to contribute also require:

* [pytest](https://pytest.org) for unit tests
* [SymPy](https://sympy.org) for the `rearrange.py` helper script.

### Installation

To install the package into your local Python environment, do:

```bash
cd <project root>
pip install -e .  # -e = editable
```

### Usage

User has to supply:

* `theta_0`: injection angle above the horizon, 0–90°
* `U_0`: injection speed [m/s]
* `s_end`: maximum value for s, i.e., the simulation limit [m]

The core logic in [simulator.py](src/waterjet_pred_valencia/simulator.py)
is designed to be easily importable into other projects.

For testing purposes, a simple CLI (command-line interface) is provided as well.
The results are saved as interactive HTML plots using bokeh.
If you've installed the package, it should be available in your environment.

```bash
waterjet-pred-valencia --angle 24 --speed 30.8 --nozzle 0.0254 --max_s 100
```

Run the command with the `-h` or `--help` option for a detailed usage description.
Console output can be activated via the debug mode (option `-d` or `--debug`).

To play around with physical and model constants
(e.g., the air entrainment rate `alpha`),
edit [parameters.py](src/waterjet_pred_valencia/parameters.py).

## To-Dos

* Fix ODE system
* Compare results to original research article
* Use as surrogate model in Smith Predictor control
