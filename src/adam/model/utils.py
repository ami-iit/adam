from pathlib import Path
from typing import Union

from adam.core.spatial_math import SpatialMath
from adam.model import MJModelFactory, ModelFactory, URDFModelFactory


def get_factory_from_string(
    model_string: str, math: SpatialMath
) -> Union[MJModelFactory, URDFModelFactory]:
    """Get the factory from the model string (path or string) and the math library

    Args:
        model_string (str): the path or the string of the model
        math (SpatialMath): the math library

    Returns:
        ModelFactory: the factory that generates the model
    """

    if Path(model_string).exists():
        model_string = Path(model_string)
        if model_string.suffix == ".xml":
            print("Loading the model from MJCF file")
            return MJModelFactory(path=model_string, math=math)
        elif model_string.suffix == ".urdf":
            print("Loading the model from URDF file")
            return URDFModelFactory(path=model_string, math=math)
        else:
            raise ValueError(
                f"The file {model_string} is not a valid MJCF or URDF file"
            )
    elif isinstance(model_string, str):
        print("Loading the model from string in URDF format")
        return URDFModelFactory(urdf_string=model_string, math=math)
    else:
        raise ValueError("The model_string is not a valid path or string")
