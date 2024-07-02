.. adam documentation master file, created by
   sphinx-quickstart on Fri Jun 28 14:10:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


adam
----

**Automatic Differentiation for rigid-body-dynamics AlgorithMs**

**adam** implements a collection of algorithms for calculating rigid-body dynamics for **floating-base** robots, in mixed and body fixed representations  using:

.. create rst list with links to the following libraries

- `Jax <https://github.com/google/jax>`_
- `CasADi <https://web.casadi.org/>`_
- `PyTorch <https://github.com/pytorch/pytorch>`_
- `NumPy <https://numpy.org/>`_


**adam** employs the automatic differentiation capabilities of these frameworks to compute, if needed, gradients, Jacobian, Hessians of rigid-body dynamics quantities. This approach enables the design of optimal control and reinforcement learning strategies in robotics.
Thanks to the `jax.vmap`-ing and `jax.jit`-ing capabilities, the algorithms can be run on batches of inputs, which are possibly converted to PyTorch using the `jax2torch` conversion functions.


**adam** is based on **Roy Featherstone's Rigid Body Dynamics Algorithms**.

Examples
--------

Have a look at the examples `folder in the repository <https://github.com/ami-iit/adam/tree/main/examples>`_!


License
-------

`BSD-3-Clause <https://choosealicense.com/licenses/bsd-3-clause/>`_


.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   installation
   quickstart/index

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules/index
