import math
import numpy as np
from numpy.typing import NDArray
from functools import wraps
from typing import Tuple, Union, List, Callable, Any

NumericDType = Union[np.float64, np.int64]

# Matrix for converting axial coordinates to pixel coordinates
axial_to_pixel_mat = np.array([[math.sqrt(3), math.sqrt(3) / 2], [0, 3 / 2.0]])

# Matrix for converting pixel coordinates to axial coordinates
pixel_to_axial_mat = np.linalg.inv(axial_to_pixel_mat)


# These are the vectors for moving from any hex to one of its neighbors.
SE = np.array((1, 0, -1))
SW = np.array((0, 1, -1))
W = np.array((-1, 1, 0))
NW = np.array((-1, 0, 1))
NE = np.array((0, -1, 1))
E = np.array((1, -1, 0))
ALL_DIRECTIONS = np.array(
    [
        NW,
        NE,
        E,
        SE,
        SW,
        W,
    ]
)


def validate_input(*validators: Callable) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for i, validator in enumerate(validators):
                if i < len(args):
                    validator(args[i])
                elif func.__code__.co_varnames[i] in kwargs:
                    validator(kwargs[func.__code__.co_varnames[i]])
                else:
                    raise ValueError(f"Missing argument for validator {i}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_numpy_array_shape(shape: tuple) -> Callable:
    def validator(arr: np.ndarray) -> None:
        if not isinstance(arr, np.ndarray) or arr.shape != shape:
            raise ValueError(f"Expected numpy array with shape {shape}")

    return validator


def validate_2d_ndarray_shape(shape: tuple) -> Callable:
    def validator(array: np.ndarray) -> None:
        for arr in array:
            if not isinstance(arr, np.ndarray) or arr.shape != shape:
                raise ValueError(f"Expected numpy array with shape {shape}")

    return validator


def validate_positive(value: float) -> None:
    if value <= 0:
        raise ValueError("Value must be positive")


def validate_not_negative(value: float) -> None:
    if value < 0:
        raise ValueError("Value can not be negative")


def validate_direction(direction: np.array) -> None:
    if direction not in ALL_DIRECTIONS:
        raise ValueError("Incorrect direction")


@validate_input(validate_not_negative)
def radius_from_hexes(hexes: int) -> NumericDType:
    """
    Computes the radius necessary to fit a set amount of hexes in the area generated
    by :func:`~hexy.get_spiral`.

    The formula documented on https://www.redblobgames.com/grids/hexagons/#rings-spiral,
    to find the amount of hexes for a given radius is::

        1 + 3 * radius * (radius+1) = s

    It is a quadratic equation that can be resolved to::

        radius = (2 * ((-s + 1) / 3)) / (-1 - sqrt(1^2 - 4 * ((-s + 1) / 3)))

    :param hexes: The amount of hexes to fit on the board.
    :type hexes: int
    :return: The radius of a spiral as an integer.
    :rtype: int
    """
    if type(hexes) != int:
        raise TypeError("radius_from_hexes() argument must be an integer, not {}".format(type(hexes)))
    return np.ceil(math.sqrt((hexes - 1) / 3 + 1 / 4) - 1 / 2)


@validate_input(validate_numpy_array_shape((3,)), validate_numpy_array_shape((3,)))
def get_cube_distance(hex_start: NDArray[NumericDType], hex_end: NDArray[NumericDType]) -> NumericDType:
    """
    Computes the smallest number of hexes between hex_start and hex_end, on the hex lattice.

    Parameters:
        hex_start (NDArray[NumericDType]): Starting hex coordinates in cube coordinate system.
        hex_end (NDArray[NumericDType]): Ending hex coordinates in cube coordinate system.

    Returns:
        NumericDType: The smallest number of hexes between `hex_start` and `hex_end`, on hex lattice.
                     This is calculated as half the sum of the absolute differences of the coordinates.
    """
    return np.sum(np.abs(hex_start - hex_end) / 2)


# Selection Functions ######


@validate_input(validate_numpy_array_shape((3,)), validate_direction)
def get_neighbor(hex: NDArray[NumericDType], direction: NDArray[NumericDType]):
    """
    Simply returns the neighbor, in the direction specified, of the hexagon.
    :param hex: Cube coordinates of the hexagon.
    :param direction: A direction from the DIR class.
    :return: The location of the neighbor in cube coordinates.
    """
    return hex + direction


@validate_input(validate_numpy_array_shape((3,)), validate_not_negative)
def get_ring(center: NDArray[NumericDType], radius: int) -> List[NDArray[NumericDType]]:
    """
    Retrieves the locations of all the hexes exactly a certain distance from a hexagon.
    :param center: The location of the hexagon to get the ring of.
    :param radius: The distance from `center` of the hexes we want.
    :return: An array of locations of the hexes that are exactly `radius` units away from `center`.
    """
    if radius < 0:
        return []
    if radius == 0:
        return [center]

    rad_hex = np.zeros((6 * radius, 3))
    count = 0
    for i in range(0, 6):
        for k in range(0, radius):
            rad_hex[count] = ALL_DIRECTIONS[i - 1] * (radius - k) + ALL_DIRECTIONS[i] * (k)
            count += 1

    return np.squeeze(rad_hex) + center


@validate_input(validate_numpy_array_shape((3,)))
def get_disk(center: NDArray[NumericDType], radius: int):
    """
    Retrieves the locations of all the hexes within `radius` hexes from a hexagon.
    :param center: The location of the hexagon to get the neighbors of.
    :param radius: The distance from `center` of the hexes we want.
    :return: An array of locations of the hexes that are within `radius` units away from `center`.
    """
    return get_spiral(center, 0, radius)


@validate_input(validate_numpy_array_shape((3,)), validate_not_negative, validate_not_negative)
def get_spiral(center: NDArray[NumericDType], radius_start: int = 1, radius_end: int = 2):
    """
    Retrieves all hexes that are between `radius_start` and `radius_end` hexes away from the `center`.
    :param center: The location of the center hex.
    :param radius_start: The distance from center. We want all hexes greater than or equal to this distance.
    :param radius_end: The distance from center. We want all hexes within this distance from `center`.
    :return: An array of locations of the hexes that are within `radius` hexes away from `center`.
    """
    hex_area = get_ring(center, radius_start)
    for i in range(radius_start + 1, radius_end + 1):
        hex_area = np.append(hex_area, get_ring(center, i), axis=0)
    return np.array(hex_area)


@validate_input(validate_numpy_array_shape((3,)), validate_numpy_array_shape((3,)))
def get_hex_line(hex_start: NDArray[NumericDType], hex_end: NDArray[NumericDType]):
    """
    Get hexes on line from hex_start to hex_end.
    :param hex_start: The hex where the line starts.
    :param hex_end: The hex where the line ends.
    :return: A set of hexes along a straight line from hex_start to hex_end.
    """
    hex_distance = get_cube_distance(hex_start, hex_end)
    if hex_distance < 1:
        return np.array([hex_start])

    # Set up linear system to compute linearly interpolated cube points
    bottom_row = np.array([i / hex_distance for i in np.arange(hex_distance)])
    x = np.vstack((1 - bottom_row, bottom_row))
    A = np.vstack((hex_start, hex_end)).T

    # linearly interpolate from a to b in n steps
    interpolated_points = A.dot(x)
    interpolated_points = np.vstack((interpolated_points.T, hex_end))
    return np.array(cube_round(interpolated_points))


# Conversion Functions ######


@validate_input(validate_2d_ndarray_shape((3,)))
def cube_to_axial(cube: NDArray[NDArray[NumericDType]]) -> NDArray[NDArray[NumericDType]]:
    """
    Convert cube to axial coordinates.
    :param cube: A coordinate in cube form. nx3
    :return: `cube` in axial form.
    """
    return np.vstack((cube[:, 0], cube[:, 2])).T


def axial_to_cube(axial: NDArray[NDArray[NumericDType]]) -> NDArray[NDArray[NumericDType]]:
    """
    Convert axial to cube coordinates.
    :param axial: A coordinate in axial form.
    :return: `axial` in cube form.
    """
    x = axial[:, 0]
    z = axial[:, 1]
    y = -x - z
    cube_coords = np.vstack((x, y, z)).T
    return cube_coords


@validate_input(validate_2d_ndarray_shape((2,)), validate_not_negative)
def axial_to_pixel(axial: NDArray[NDArray[NumericDType]], radius: int) -> NDArray[NDArray[NumericDType]]:
    """
    Converts the location of a hex in axial form to pixel coordinates.
    :param axial: The location of a hex in axial form. nx3
    :param radius: Radius of all hexagons.
    :return: `axial` in pixel coordinates.
    """
    pos = radius * axial_to_pixel_mat.dot(axial.T)
    return pos.T


@validate_input(validate_2d_ndarray_shape((3,)), validate_not_negative)
def cube_to_pixel(cube: NDArray[NDArray[NumericDType]], radius: int):
    """
    Converts the location of a hex in cube form to pixel coordinates.
    :param cube: The location of a hex in cube form. nx3
    :param radius: Radius of all hexagons.
    :return: `cube` in pixel coordinates.
    """
    in_axial_form = cube_to_axial(cube)
    return axial_to_pixel(in_axial_form, radius)


@validate_input(validate_2d_ndarray_shape((2,)), validate_not_negative)
def pixel_to_cube(pixel: NDArray[NDArray[NumericDType]], radius: int):
    """
    Converts the location of a hex in pixel coordinates to cube form.
    :param pixel: The location of a hex in pixel coordinates. nx2
    :param radius: Radius of all hexagons.
    :return: `pixel` in cube coordinates.
    """
    axial = pixel_to_axial_mat.dot(pixel.T) / radius
    return cube_round(axial_to_cube(axial.T))


@validate_input(validate_2d_ndarray_shape((2,)), validate_not_negative)
def pixel_to_axial(pixel: NDArray[NDArray[NumericDType]], radius: int):
    """
    Converts the location of a hex in pixel coordinates to axial form.
    :param pixel: The location of a hex in pixel coordinates. nx2
    :param radius: Radius of all hexagons.
    :return: `pixel` in axial coordinates.
    """
    cube = pixel_to_cube(pixel, radius)
    return cube_to_axial(cube)


@validate_input(validate_2d_ndarray_shape((3,)))
def cube_round(cubes: NDArray[NDArray[NumericDType]]):
    """
    Rounds a location in cube coordinates to the center of the nearest hex.
    :param cubes: Locations in cube form. nx3
    :return: The location of the center of the nearest hex in cube coordinates.
    """
    rounded = np.zeros((cubes.shape[0], 3))
    rounded_cubes = np.round(cubes)
    for i, cube in enumerate(rounded_cubes):
        (rx, ry, rz) = cube
        xdiff, ydiff, zdiff = np.abs(cube - cubes[i])
        if xdiff > ydiff and xdiff > zdiff:
            rx = -ry - rz
        elif ydiff > zdiff:
            ry = -rx - rz
        else:
            rz = -rx - ry
        rounded[i] = (rx, ry, rz)
    return rounded


@validate_input(validate_2d_ndarray_shape((2,)))
def axial_round(axial: NDArray[NDArray[NumericDType]]):
    """
    Rounds a location in axial coordinates to the center of the nearest hex.
    :param axial: A location in axial form. nx2
    :return: The location of the center of the nearest hex in axial coordinates.
    """
    return cube_to_axial(cube_round(axial_to_cube(axial)))
