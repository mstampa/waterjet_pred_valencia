# Fire stream trajectory model — Valencia et al.

* [Prerequisites](#prerequisites)
* [Usage](#usage)
* [To-Dos](#to-dos)

**NOTE:** Work in progress and currently not functional!
There seem to be errors in the ODE system. Example plots were generated with some of the computations bypassed.

This is a Python module for simulating the trajectories of "fire streams", i.e.,
large water jets shot from fire monitors. The backbone is a 1D Eulerian analytical
model reproduced from this publication, with the author's kind support:

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

The model describes the fire stream as a combination of three phases: water-core phase,
air phase, and spray phase. It includes air entrainment, jet break-up spray generation
and multi-dispersion (i.e., droplets of multiple sizes).
Trajectory and spray behavior are modeled over the streamwise axis "s".

The simulation stops if:
* the limit for s (simulation span) is reached
* y <= 0 (the stream hits the ground)
* assert statements catch a physical impossibility (e.g., negative diameter)

Example plots produced with default parameters:

![Example plots](doc/example_plots.png)

## Prerequisites

Key dependencies of this package that need to be installed on your system / in your venv:

* [SciPy](https://www.scipy.org) for solving the ODE
* [bokeh](https://www.bokeh.org) for plotting

To get the equations from the paper into the form expected by SciPy's `solve_ivp()` function, the `rearrange.py` helpers script uses [SymPy](https://www.sympy.org).

## Usage

User has to supply:
- Injection angle above the horizon (between 0 and 90°)
- Injection velocity [m/s]
- simulation span (maximum value for "s", use an upper limit of the trajectory's length)

The core library is designed to be easily importable into other projects.
For testing purposes, a simple **CLI** is provided. Example run:

```bash
cd waterjet_pred_valencia
./run_simulation.py -a 24 -s 30.8 -n 0.0254 -l 100
``` 

Run the command with the `-h` or `--help` option for a detailed usage description.

To play around with physical and model constants (e.g., the air entrainment rate alpha),
edit `model/parameters.py`.

## To-Dos

* Fix ODE system
* Compare results to original research article
* Proper packaging
* Include as module in other projects
