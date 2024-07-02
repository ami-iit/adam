PyTorch Batched interface
=========================

This module implements the batched version of the Rigid Body Dynamics algorithms using PyTorch.
This module uses Jax under the hood, then the functions are ``jax.vmap``-ed and ``jax.jit``-ed to run on batches of inputs, which are ultimately converted to PyTorch using the ``jax2torch`` conversion functions.

.. note::

   The first time you run a function from this module, it will take a bit longer to execute as they are being compiled by Jax.


.. note::

   If the GPU support for ``JAX`` is needed, follow the instructions in the `Jax documentation <https://jax.readthedocs.io/en/latest/installation.html#conda-community-supported>`_.


----------------

.. currentmodule:: adam.pytorch.computation_batch

.. automodule:: adam.pytorch.computation_batch
   :members:
   :undoc-members:
   :show-inheritance:
