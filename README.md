# Fire stream trajectory model — Valencia et al.

**NOTE:** Work in progress and currently not functional!
There seem to be errors in the ODE system.
Example plots were generated with some of the computations bypassed.

This is a Python module for simulating the trajectories of "fire streams", i.e.,
large water jets shot from fire monitors.
The backbone is a 1D Eulerian analytical
model reproduced from this publication:

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

![Example plots](doc/example_plots.png)

## Prerequisites

Key dependencies of this package are:
* SciPy for solving the ODE
* bokeh for plotting

To get the equations from the paper into the form expected by SciPy's `solve_ivp()` function, the `rearrange.py` script used SymPy.

## Usage

User has to supply:
- theta_0: injection angle above the horizon in range (0°, 90°)
- U_0: injection speed [m/s]
- s_end: maximum value for s, aka the simulation limit [m]

The core logic, found in the [model](model) folder, is designed to be easily importable into other projects.

For testing purposes, a simple **CLI** is provided as well.
It calls `model.simulator.simulate()` to simulate a fire stream trajectory with the given arguments.
The results are saved as interactive HTML plots using bokeh.

Example run:

``` 
python waterjet_pred_valencia.py --angle 24 --speed 30.8 --nozzle 0.0254 --max_s 100
``` 

Run the command with the `-h` or `--help` option for a detailed usage description.
Console output can be activated via the debug mode (option `-d`).

To play around with physical and model constants (e.g., the air entrainment rate alpha),
edit [model/parameters.py](model/parameters.py).

## To-Dos

* Fix ODE system
* Compare results to original research article
* Proper Python packaging
* Use as surrogate model in Smith Predictor control
