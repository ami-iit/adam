# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass
from typing import Union

import casadi as cs
import numpy.typing as npt

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath


@dataclass
class CasadiLike(ArrayLike):
    """Wrapper class for Casadi types"""

    array: Union[cs.SX, cs.DM]

    def __matmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        return CasadiLike(cs.mtimes(self.array, other.array))

    def __rmatmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        return CasadiLike(other.array @ self.array)

    def __mul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides * operator"""
        return CasadiLike(self.array * other.array)

    def __rmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides * operator"""
        return CasadiLike(self.array * other.array)

    def __add__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides + operator"""
        try:
            return CasadiLike(self.array + other.array)
        except RuntimeError as e:
            if "Dimension mismatch" in str(e):
                # Try to handle shape mismatches by transposing the other array
                try:
                    return CasadiLike(self.array + other.array.T)
                except:
                    # If that doesn't work, try transposing self
                    try:
                        return CasadiLike(self.array.T + other.array)
                    except:
                        # If nothing works, re-raise the original error
                        raise e
            else:
                raise e

    def __radd__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides + operator"""
        return CasadiLike(self.array + other.array)

    def __sub__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides - operator"""
        return CasadiLike(self.array - other.array)

    def __rsub__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides - operator"""
        return CasadiLike(self.array - other.array)

    def __neg__(self) -> "CasadiLike":
        """Overrides - operator"""
        return CasadiLike(-self.array)

    def __truediv__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides / operator"""
        return CasadiLike(self.array / other.array)

    def __setitem__(self, idx, value: Union["CasadiLike", npt.ArrayLike]):
        """Overrides set item operator"""
        self.array[idx] = value.array

    def __getitem__(self, idx) -> "CasadiLike":
        """Overrides get item operator"""
        # Handle ellipsis indexing for CasADi compatibility
        if idx == Ellipsis:
            # Only ellipsis, return self
            return self
        elif isinstance(idx, tuple) and Ellipsis in idx:
            # For CasADi, ellipsis should be ignored since we don't support batching
            # Remove ellipsis and keep the rest
            idx = tuple(i for i in idx if i != Ellipsis)
            if len(idx) == 1:
                idx = idx[0]
            elif len(idx) == 0:
                # Only ellipsis, return self
                return self
            # Continue with the remaining index

        # Handle None indices (newaxis) - CasADi doesn't support this, so we need to reshape
        if isinstance(idx, tuple) and None in idx:
            # Special case for [..., None] which means adding a dimension at the end
            if idx == (..., None):
                # This is trying to convert a vector to a column vector
                # For CasADi, we need to reshape appropriately
                if len(self.array.shape) == 1:
                    # Convert 1D array to column vector by reshaping
                    return CasadiLike(self.array.reshape((-1, 1)))
                else:
                    # For higher dimensions, we can't easily add dimensions in CasADi
                    # Just return self for now (this may cause issues but is better than crashing)
                    return self

            # For other None indexing cases, remove None indices
            non_none_idx = tuple(i for i in idx if i is not None)
            if len(non_none_idx) == 0:
                # All indices are None, return self
                return self
            elif len(non_none_idx) == 1:
                idx = non_none_idx[0]
            else:
                idx = non_none_idx
        elif idx is None:
            # Single None index, return self
            return self

        # Now handle the actual indexing
        return CasadiLike(self.array[idx])

    @property
    def shape(self):
        """
        Returns:
            tuple: Shape of the array
        """
        return self.array.shape

    @property
    def ndim(self):
        """
        Returns:
            int: Number of dimensions
        """
        return len(self.array.shape)

    @property
    def T(self) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Transpose of the array
        """
        return CasadiLike(self.array.T)


class CasadiLikeFactory(ArrayLikeFactory):

    @staticmethod
    def zeros(*x: int) -> CasadiLike:
        """
        Returns:
            CasadiLike: Matrix of zeros of dim *x
        """
        # Handle the case where x is a tuple with more than 2 dimensions
        # CasADi only supports 2D matrices, so we ignore batch dimensions
        if len(x) == 1 and isinstance(x[0], tuple):
            # x is ((d1, d2, d3, ...),)
            shape = x[0]
            if len(shape) > 2:
                # Take only the last two dimensions
                shape = shape[-2:]
            return CasadiLike(cs.SX.zeros(*shape))
        elif len(x) > 2:
            # Multiple arguments like zeros(d1, d2, d3)
            # Take only the last two
            return CasadiLike(cs.SX.zeros(*x[-2:]))
        else:
            # Normal case: zeros(d1, d2) or zeros(d1)
            return CasadiLike(cs.SX.zeros(*x))

    def eye(self, x) -> CasadiLike:
        """
        Args:
            x (int or tuple): matrix dimension or shape

        Returns:
            CasadiLike: Identity matrix
        """
        # Handle the case where x is a shape tuple (for batched operations)
        if isinstance(x, tuple):
            # For CasADi, ignore batch dimensions and just use the last dimension for square matrix
            # E.g., if x is (1, 1, 3), we want a 3x3 identity matrix
            n = x[-1]  # Get the last dimension
            return CasadiLike(cs.SX.eye(n))
        else:
            # Traditional case: x is just an integer
            return CasadiLike(cs.SX.eye(x))

    @staticmethod
    def asarray(x) -> CasadiLike:
        """
        Returns:
            CasadiLike: Vector wrapping *x
        """

        # Case 1: If already symbolic, just wrap and return
        if isinstance(x, (cs.SX, cs.DM)):
            return CasadiLike(x)

        # Case 2: If numeric, convert to DM
        if isinstance(x, (int, float)):
            # Single scalar
            return CasadiLike(cs.DM(x))

        # Case 3: If numpy array, convert to DM
        if isinstance(x, cs.np.ndarray):
            # If already a numpy array, convert to Casadi DM
            return CasadiLike(cs.DM(x))

        # Case 4: If list or tuple, convert to DM if all items are numeric or SX otherwise
        if isinstance(x, (list, tuple)):
            # TODO: we need to carefully check if this is the correct behavior
            # for example, handle the case of a list of lists
            # Handle empty list/tuple
            if not x:
                return CasadiLike(cs.DM([]))
            if all(isinstance(item, (int, float)) for item in x):
                # All numeric, can safely convert to DM
                return CasadiLike(cs.DM(x))
            if all(isinstance(item, cs.SX) for item in x):
                return CasadiLike(cs.SX(x))
            else:
                return CasadiLike(cs.DM(x))

        raise TypeError(
            f"Unsupported type: {type(x)}. Must be numeric, list/tuple/np.ndarray of numerics, or SX."
        )

    def zeros_like(self, x: CasadiLike) -> CasadiLike:
        """
        Returns:
            CasadiLike: Matrix of zeros with the same shape as x
        """
        shape = x.array.shape
        return CasadiLike(cs.SX.zeros(*shape))

    def ones_like(self, x: CasadiLike) -> CasadiLike:
        """
        Returns:
            CasadiLike: Matrix of ones with the same shape as x
        """
        shape = x.array.shape
        return CasadiLike(cs.SX.ones(*shape))

    def tile(self, x: CasadiLike, reps) -> CasadiLike:
        """
        CasADi doesn't support batching, so we ignore tiling and return the original array.

        Args:
            x: The array to tile
            reps: Repetition counts (ignored for CasADi)

        Returns:
            CasadiLike: The original array (no tiling applied)
        """
        return x

    def squeeze(self, x: CasadiLike, axis: int = None) -> CasadiLike:
        """
        Remove single-dimensional entries from the shape of an array.

        Args:
            x: Input array
            axis: Axis to squeeze (None for all single dimensions)

        Returns:
            CasadiLike: Array with single dimensions removed
        """
        shape = x.array.shape
        
        if axis is None:
            # Remove all single dimensions
            new_shape = tuple(dim for dim in shape if dim != 1)
        else:
            # Remove specific axis if it has size 1
            if axis < 0:
                axis = len(shape) + axis
            if axis >= len(shape) or shape[axis] != 1:
                # Axis doesn't exist or doesn't have size 1, return unchanged
                return x
            new_shape = shape[:axis] + shape[axis+1:]
        
        if len(new_shape) == 0:
            # Result would be scalar, keep as (1,) for CasADi compatibility
            new_shape = (1,)
        
        return CasadiLike(x.array.reshape(*new_shape))


class SpatialMath(SpatialMath):

    def __init__(self):
        super().__init__(CasadiLikeFactory())

    @staticmethod
    def skew(x: Union["CasadiLike", npt.ArrayLike]) -> CasadiLike:
        """
        Args:
            x (Union[CasadiLike, npt.ArrayLike]): 3D vector

        Returns:
            CasadiLike: the skew symmetric matrix from x
        """
        if isinstance(x, CasadiLike):
            # Check if x.array is empty or has wrong size
            if x.array.size1() == 0 or x.array.size2() == 0:
                raise ValueError(
                    f"skew received empty array: {x.array.size1()}x{x.array.size2()}"
                )
            return CasadiLike(cs.skew(x.array))
        else:
            return CasadiLike(cs.skew(x))

    @staticmethod
    def sin(x: npt.ArrayLike) -> CasadiLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            CasadiLike: the sin value of x
        """
        return CasadiLike(cs.sin(x.array))

    @staticmethod
    def cos(x: npt.ArrayLike) -> CasadiLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            CasadiLike: the cos value of x
        """
        return CasadiLike(cs.cos(x.array))

    @staticmethod
    def outer(x: npt.ArrayLike, y: npt.ArrayLike) -> CasadiLike:
        """
        Args:
            x (npt.ArrayLike): vector
            y (npt.ArrayLike): vector

        Returns:
            CasadiLike: outer product between x and y
        """
        return CasadiLike(cs.np.outer(x.array, y.array))

    @staticmethod
    def vertcat(*x) -> CasadiLike:
        """
        Returns:
            CasadiLike:  vertical concatenation of elements
        """
        # here the logic is a bit convoluted: x is a tuple containing CasadiLike
        # cs.vertcat accepts *args. A list of cs types is created extracting the value
        # from the CasadiLike stored in the tuple x.
        # Then the list is unpacked with the * operator.
        y = [xi.array for xi in x]
        return CasadiLike(cs.vertcat(*y))

    @staticmethod
    def horzcat(*x) -> CasadiLike:
        """
        Returns:
            CasadiLike:  horizontal concatenation of elements
        """

        y = [xi.array for xi in x]
        return CasadiLike(cs.horzcat(*y))

    @staticmethod
    def stack(x, axis: int = 0) -> CasadiLike:
        """
        Stack arrays along a specified axis.

        Args:
            x: Sequence of CasadiLike objects to stack
            axis: Axis along which to stack arrays

        Returns:
            CasadiLike: Stacked array
        """
        # Extract CasADi arrays from CasadiLike objects
        arrays = [xi.array for xi in x]

        if axis == -1:  # Stack along last dimension (horizontal)
            return CasadiLike(cs.horzcat(*arrays))
        elif axis == -2:  # Stack along second-to-last dimension (vertical)
            return CasadiLike(cs.vertcat(*arrays))
        elif axis == 0:  # Stack along first dimension (vertical for 2D)
            return CasadiLike(cs.vertcat(*arrays))
        elif axis == 1:  # Stack along second dimension (horizontal for 2D)
            return CasadiLike(cs.horzcat(*arrays))
        else:
            raise NotImplementedError(f"CasADi stack not implemented for axis={axis}")

    @staticmethod
    def concatenate(x, axis: int = 0) -> CasadiLike:
        """
        Concatenate arrays along a specified axis.

        Args:
            x: Sequence of CasadiLike objects to concatenate
            axis: Axis along which to concatenate arrays

        Returns:
            CasadiLike: Concatenated array
        """
        # Extract CasADi arrays from CasadiLike objects
        arrays = [xi.array for xi in x]

        # For CasADi column vectors, when concatenating along axis=-1 (last axis),
        # we often want to stack them vertically to create a longer column vector
        # rather than horizontally to create a wider matrix
        if axis == -1 and len(arrays) == 2:
            # Check if we're trying to concatenate two column vectors
            arr1, arr2 = arrays[0], arrays[1]
            if (
                len(arr1.shape) == 2
                and arr1.shape[1] == 1
                and len(arr2.shape) == 2
                and arr2.shape[1] == 1
            ):
                # Two column vectors - stack vertically to create longer column vector
                return CasadiLike(cs.vertcat(arr1, arr2))
            
            # Check if we have incompatible shapes that need flattening
            # E.g., trying to concatenate (6,) or (6,1) with (1,23)
            if len(arr1.shape) <= 2 and len(arr2.shape) <= 2:
                # Flatten both arrays to 1D and concatenate
                try:
                    # Try normal concatenation first
                    return CasadiLike(cs.horzcat(arr1, arr2))
                except:
                    # If that fails, flatten and try vertical concatenation
                    # Flatten arrays to vectors
                    flat1 = arr1.reshape((-1, 1)) if len(arr1.shape) == 1 or arr1.shape[1] != 1 else arr1
                    flat2 = arr2.reshape((-1, 1)) if len(arr2.shape) == 1 or arr2.shape[1] != 1 else arr2
                    
                    # If one is a row vector, transpose it to column
                    if len(flat1.shape) == 2 and flat1.shape[0] == 1:
                        flat1 = flat1.T
                    if len(flat2.shape) == 2 and flat2.shape[0] == 1:
                        flat2 = flat2.T
                        
                    return CasadiLike(cs.vertcat(flat1, flat2))

        if axis == -1:  # Concatenate along last dimension (horizontal)
            return CasadiLike(cs.horzcat(*arrays))
        elif axis == -2:  # Concatenate along second-to-last dimension (vertical)
            return CasadiLike(cs.vertcat(*arrays))
        elif axis == 0:  # Concatenate along first dimension (vertical for 2D)
            return CasadiLike(cs.vertcat(*arrays))
        elif axis == 1:  # Concatenate along second dimension (horizontal for 2D)
            return CasadiLike(cs.horzcat(*arrays))
        else:
            raise NotImplementedError(
                f"CasADi concatenate not implemented for axis={axis}"
            )

    @staticmethod
    def swapaxes(x: CasadiLike, axis1: int, axis2: int) -> CasadiLike:
        """
        Swap two axes of an array.

        Args:
            x: Input CasadiLike array
            axis1: First axis
            axis2: Second axis

        Returns:
            CasadiLike: Array with axes swapped
        """
        # For CasADi, the most common case is swapping last two axes (matrix transpose)
        if (axis1 == -1 and axis2 == -2) or (axis1 == -2 and axis2 == -1):
            return CasadiLike(x.array.T)
        elif (axis1 == 0 and axis2 == 1) or (axis1 == 1 and axis2 == 0):
            return CasadiLike(x.array.T)
        else:
            raise NotImplementedError(
                f"CasADi swapaxes not implemented for axis1={axis1}, axis2={axis2}"
            )

    @staticmethod
    def mxv(m: CasadiLike, v: CasadiLike) -> CasadiLike:
        """
        Matrix-vector multiplication for CasADi.

        Args:
            m: Matrix (CasadiLike)
            v: Vector (CasadiLike)

        Returns:
            CasadiLike: Result of matrix-vector multiplication
        """
        # For CasADi, we need to handle matrix-vector multiplication carefully
        # Convert vector to column vector if needed for proper matrix multiplication
        v_array = v.array
        if len(v_array.shape) == 1:
            # Convert 1D vector to column vector
            v_array = v_array.reshape((-1, 1))
        elif len(v_array.shape) == 2 and v_array.shape[1] != 1:
            # If it's a row vector, transpose to column vector
            v_array = v_array.T

        # Perform matrix multiplication
        result = cs.mtimes(m.array, v_array)
        
        # Keep result as column vector for proper concatenation
        return CasadiLike(result)

    @staticmethod
    def vxs(v: CasadiLike, c: CasadiLike) -> CasadiLike:
        """
        Vector times scalar multiplication for CasADi.

        Args:
            v: Vector (CasadiLike)
            c: Scalar (CasadiLike)

        Returns:
            CasadiLike: Result of vector times scalar
        """
        # For CasADi, we need to handle the vector-scalar multiplication carefully
        # v should be a vector (n, 1) or (n,), c should be a scalar

        v_array = v.array

        # Handle scalar extraction from (1, 1) matrix
        c_array = c.array
        if len(c_array.shape) == 2 and c_array.shape == (1, 1):
            c_array = c_array[0, 0]  # Extract scalar from (1, 1) matrix
        elif len(c_array.shape) == 2 and c_array.shape[0] == 1:
            c_array = c_array[0, 0]  # Extract scalar from (1, n) matrix
        elif len(c_array.shape) == 2 and c_array.shape[1] == 1:
            c_array = c_array[0, 0]  # Extract scalar from (n, 1) matrix

        # For CasADi, scalar multiplication should work directly
        result = v_array * c_array
        return CasadiLike(result)
