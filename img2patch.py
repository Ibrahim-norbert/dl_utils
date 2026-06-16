# =============================================================================
# Import third-party libraries
import numpy as np
from numpy import ndarray
from pathlib import Path
from typing import Union, Tuple
from skimage import io, transform, exposure
from skimage.measure import regionprops_table


# Create a bbox class
class BBoxes:
    """Class to calculate and represent bounding boxes from a mask file"""
    bbox: np.ndarray

    # Constructor
    def __init__(self, bboxes, mask=None, image=None) -> None:
        """

        :type bboxes: np.ndarray
        :type mask: np.ndarray
        :param bboxes: A numpy array with the bounding boxes.
        :param mask: A numpy array with the mask.
        :return: Object of type BBoxes.
        """
        # Read in the mask file
        self.mask = mask
        self.bboxes = bboxes
        self.image = image

        if isinstance(image, str):
            self.img_path = image
        if isinstance(image, np.ndarray):
            self.image = image

    # TODO: Implement a class method to create a BBoxes object from a mask array. (outside file loading)
    @classmethod
    def from_mask(cls,
                  mask: np.ndarray,
                  image: object = None) -> object:
        """
        Calculates the bounding boxes from the mask file.
        :param mask: A numpy array with the mask.
        :param image: A numpy array with the image.
        :return: Object of type BBoxes.
        """
        if mask.ndim < 3:
            raise ValueError(f"Mask must be a 3D volume (z, y, x); got ndim={mask.ndim}.")
        # Add check if mask contains any elements
        if np.max(mask) == 0:
            raise ValueError("Mask contains no elements.")

        # Get indexes of nonzero elements
        props = regionprops_table(mask, properties=('label', 'bbox'))

        if not props['label'].tolist():
            print("No coordinates for non zero elements could be deduced with 'np.nonzero'")

        # Get the bounding boxes using largest diameter for each identity
        # The order is z,y,x
        bboxes = np.array([props['label'], props['bbox-0'], props['bbox-3'],
                           props['bbox-1'], props['bbox-4'], props['bbox-2'], props['bbox-5']]).T

        return cls(bboxes, mask, image)

    # Column layout of a bbox row: [id, z_min, z_max, y_min, y_max, x_min, x_max].
    _MIN_COLS = [1, 3, 5]   # z_min, y_min, x_min
    _MAX_COLS = [2, 4, 6]   # z_max, y_max, x_max

    @staticmethod
    def iou(box1: np.ndarray,
            box2: np.ndarray) -> float:
        """
        Calculates the volumetric (3D) IoU for two bounding boxes.
        :param box1: Numpy array with the first bounding box.
        :param box2: Numpy array with the second bounding box.
        :return: A float with the IoU value between the two bounding boxes.
        """
        lo = np.maximum(box1[BBoxes._MIN_COLS], box2[BBoxes._MIN_COLS])
        hi = np.minimum(box1[BBoxes._MAX_COLS], box2[BBoxes._MAX_COLS])

        intersection = np.prod(np.clip(hi - lo, 0, None))
        if intersection == 0:
            return 0.0

        vol1 = np.prod(box1[BBoxes._MAX_COLS] - box1[BBoxes._MIN_COLS])
        vol2 = np.prod(box2[BBoxes._MAX_COLS] - box2[BBoxes._MIN_COLS])
        return float(intersection / (vol1 + vol2 - intersection))

    @property
    def iou_matrix(self) -> np.ndarray:
        """
        Returns the upper-triangular pairwise (3D) IoU matrix for the bounding boxes.
        :return: An (n, n) numpy array with the IoU values for each bounding box pair.
        """
        mins = self.bboxes[:, self._MIN_COLS].astype(float)   # (n, 3)
        maxs = self.bboxes[:, self._MAX_COLS].astype(float)    # (n, 3)

        # Pairwise overlap extents via broadcasting: (n, n, 3).
        lo = np.maximum(mins[:, None, :], mins[None, :, :])
        hi = np.minimum(maxs[:, None, :], maxs[None, :, :])
        intersection = np.prod(np.clip(hi - lo, 0, None), axis=-1)  # (n, n)

        volumes = np.prod(maxs - mins, axis=-1)                    # (n,)
        union = volumes[:, None] + volumes[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, intersection / union, 0.0)

        # Keep only the strict upper triangle, matching the previous behaviour.
        return np.triu(iou, k=1)

    # Bounding box IoU operations
    def are_overlapping(self) -> np.array:
        """
        Returns a boolean array indicating whether the bounding boxes are overlapping.
        :return: A numpy array with boolean values for each bounding box pair.
        """
        # Get the IoU matrix
        overlapping = np.where(self.iou_matrix > 0)

        # Get the identities from the overlapping indexes
        identities = self.bboxes[np.unique(overlapping), 0]

        return identities

    # Magic methods
    def __len__(self) -> int:
        return self.bboxes.shape[0]

    def __getitem__(self, item) -> ndarray:
        """
        Returns the bounding box at the given index. The index starts at 1.
        No negative indexing is allowed.

        :param item: Index of the bounding box to return.
        :return: Numpy array with the bounding box.
        """
        if item <= 0:
            raise IndexError("Index must be greater than 0, use the mask id to get the bounding box. "
                             "(i.e. mask_id = 1 recovers the bounding box of the first cell in the mask file.)")

        return self.bboxes[item - 1]

    def __str__(self):
        return str(self.bboxes)

    def __setattr__(self, key: str, value: object) -> None:
        if key == 'image':
            if isinstance(value, np.ndarray):
                self.__dict__[key] = value
            elif isinstance(value, str) or isinstance(value, Path):
                self.__dict__[key] = io.imread(value)
            elif value is None:
                self.__dict__[key] = value
            else:
                raise TypeError("Image must be of type numpy array or string.")

        elif key == 'mask':
            if isinstance(value, np.ndarray):
                self.__dict__[key] = value
            elif isinstance(value, str) or isinstance(value, Path):
                self.__dict__[key] = io.imread(value)
            elif value is None:
                self.__dict__[key] = value
            else:
                raise TypeError("Mask must be of type numpy array or string.")

        else:
            self.__dict__[key] = value

    def sample(self,
               n: int = 5) -> object:
        """
        Randomly selects n bounding boxes from the object.
        :param n: Number of bounding boxes to select [default=5].
        :return: Object of type BBoxes with the selected bounding boxes.
        """
        # Randomly select n bounding boxes
        idx = np.random.choice(self.bboxes.shape[0], n, replace=False)

        return BBoxes(self.bboxes[idx], self.mask, self.image)

    # Bounding box operations
    def expand(self,
               n: int = 0) -> object:
        """
        Expands the bounding boxes by n pixels.
        :param n: Integer with the number of pixels to expand the bounding boxes.
        :return: Object of type BBoxes with the expanded bounding boxes.
        """
        # Expand the bounding boxes by n pixels, but not beyond the image size.
        z, y, x = self.mask.shape
        expanded = self.bboxes.copy()
        expanded[:, 1] = np.clip(self.bboxes[:, 1] - n, 0, None)        # z_min
        expanded[:, 2] = np.clip(self.bboxes[:, 2] + n, None, z)        # z_max
        expanded[:, 3] = np.clip(self.bboxes[:, 3] - n, 0, None)        # y_min
        expanded[:, 4] = np.clip(self.bboxes[:, 4] + n, None, y)        # y_max
        expanded[:, 5] = np.clip(self.bboxes[:, 5] - n, 0, None)        # x_min
        expanded[:, 6] = np.clip(self.bboxes[:, 6] + n, None, x)        # x_max

        return BBoxes(expanded, self.mask, self.image)

    def DynamicPadding(self,
               expected: np.ndarray) -> object:
        """
        Pad each bounding box so it reaches the ``expected`` size.

        :param expected: 1D array of length ``len(self)`` giving, per box, the
            target size; each box is grown symmetrically by ``(expected - size) / 2``
            on every axis, clamped to the image bounds.
        :return: Object of type BBoxes with the padded bounding boxes.
        """
        # Current extents per axis: (z_max - z_min, y_max - y_min, x_max - x_min).
        sizes = self.bboxes[:, [2, 4, 6]] - self.bboxes[:, [1, 3, 5]]
        # Per-box, per-axis padding to add on each side.
        ns = ((expected[:, None] - sizes) / 2).astype(int)

        assert self.bboxes.shape[0] == ns.shape[0], "Must have same length"

        expanded = np.stack([np.array([x[0],
                                max(x[1] - n[0], 0),
                             min(x[2] + n[0], self.mask.shape[0]),
                             max(x[3] - n[1], 0),
                             min(x[4] + n[1], self.mask.shape[1]),
                             max(x[5] - n[2], 0),
                             min(x[6] + n[2], self.mask.shape[2])]) for x, n in zip(self.bboxes, ns) ], axis=0)

        assert expanded.shape == self.bboxes.shape, "Output does not have same shape"

        return BBoxes(expanded, self.mask, self.image).remove_from_edge()

    def identities(self) -> np.array:
        """
        Returns the identities of the bounding boxes.
        :return: numpy array with the identities of the bounding boxes.
        """
        return self.bboxes[:, 0]

    def idx(self) -> np.array:
        """
        Returns the indexes in base 0 for the bounding boxes.
        :return: numpy array with the indexes of the bounding boxes.
        """
        return self.bboxes[:, 0] - 1

    # Bounding box properties
    def get_sides(self) -> np.array:
        """
        Returns the sides of the bounding boxes.
        :return: numpy array with the sides of the bounding boxes.
        """
        # Get the sides of the bounding boxes
        return np.array([self.bboxes[:, 0],
                         self.bboxes[:, 2] - self.bboxes[:, 1],
                         self.bboxes[:, 4] - self.bboxes[:, 3],
                         self.bboxes[:, 6] - self.bboxes[:, 5]]).T

    def get_volume(self) -> np.ndarray:
        """
        Returns the areas of the bounding boxes.
        :return: numpy array with the areas of the bounding boxes.
        """
        # Get the areas of the bounding boxes
        return np.array([self.bboxes[:, 0],
                         (self.bboxes[:, 2] - self.bboxes[:, 1]) *
                         (self.bboxes[:, 4] - self.bboxes[:, 3]) * (self.bboxes[:, 6] - self.bboxes[:, 5])]).T

    def get_ratios(self) -> np.ndarray:
        """
        Returns the aspect ratios of the bounding boxes.
        :return: numpy array with the aspect ratios of the bounding boxes.
        """
        # Get the aspect ratios of the bounding boxes
        ratios = np.array((self.bboxes[:, 2] - self.bboxes[:, 1]) / (self.bboxes[:, 4] - self.bboxes[:, 3]))

        return np.array(([self.bboxes[:, 0], ratios]))

    def get_centers(self) -> np.ndarray:
        """
        Returns the centers of the bounding boxes.
        :return: numpy array with the centers of the bounding boxes.
        """

        # Get the centers of the bounding boxes
        values = np.stack([self.bboxes[:, 0],
                         np.floor((self.bboxes[:, -6] + self.bboxes[:, -5]) / 2).astype(int),
                         np.floor((self.bboxes[:, -4] + self.bboxes[:, -3]) / 2).astype(int),
                         np.floor((self.bboxes[:, -2] + self.bboxes[:, -1]) / 2).astype(int)], axis=-1)

        return values

    def get(self,
            value: str = "area") -> np.ndarray:
        """
        Returns the values of the bounding boxes based on the given parameter.
        :param value: Mode to use for getting the values of the bounding boxes [default="area"].
        :return:
        """
        # Get the values of the bounding boxes
        if value == 'area':
            values = self.get_volume()
        elif value == 'ratio':
            values = self.get_ratios()
        elif value == 'center':
            values = self.get_centers()
        elif value == 'sides':
            values = self.get_sides()
        else:
            raise NotImplementedError('Invalid filter parameter, please select from area, ratio, center or sides.')

        return values

    # Get overlapping pairs
    def get_overlapping_pairs(self) -> (np.array, np.array):
        """
        Returns the overlapping pairs of bounding boxes. The first array contains the identities, the second the IoU
        values.
        :return: A tuple of two numpy arrays.
        """
        # Get the IoU matrix
        iou_matrix = self.iou_matrix

        # Get elements that are not zero
        x, y = np.where(iou_matrix > 0)
        v = iou_matrix[x, y ]
        x = self.bboxes[x, 0]
        y = self.bboxes[y, 0]

        return np.array([x, y]).T, v

    def subset(self,
               indexes: np.ndarray) -> object:
        """
        Subset boxes from the BBox object.
        :param indexes: Numpy array with the cellIDs/maskIDs to subset the bounding boxes.
        :return: Object of type BBoxes with the subset bounding boxes.
        """
        # Find indices where the values in the first column match the filter_array
        return BBoxes(self.bboxes[np.isin(self.bboxes[:, 0], indexes)], self.mask, self.image)

    # Bounding box filters
    def filter(self,
               by: str = "area",
               operator: np.ufunc = np.greater_equal,
               value: Union[float, Tuple[float, float]] = 0) -> object:
        """
        Filter the bounding boxes based on the given parameters
        :param by: Choose between area, ratio, center or dims to filter the bounding boxes [default="area"].
        :param operator: Numpy comparison operator to use [default=np.greater_equal]
        :param value: Value to be used for filtering [default=0].
        :return: np.ndarray
        """

        # Get the values of the bounding boxes that are filtered
        values = self.get(by)

        # Filter the bounding boxes based on the given parameters
        if by == "sides":
            idx = np.where(operator(values[:, 1], value[0]) & operator(values[:, 2], value[1]) & operator(values[:, 3], value[2]))[0]
        else:
            idx = np.where(operator(values[:, 1], value))[0]

        return BBoxes(self.bboxes[idx], self.mask, self.image)

    def remove_from_edge(self) -> object:
        """
        Removes the bounding boxes that are on the edge of the image.
        :return: BBoxes object with the bounding boxes that are not on the edge of the image.
        """
        # Removes the bounding boxes that are on the edge of the image
        idx = np.where((self.bboxes[:, 1] >= 0) &
                       (self.bboxes[:, 2] < self.mask.shape[0]) &
                       (self.bboxes[:, 3] > 0) &
                       (self.bboxes[:, 4] < self.mask.shape[1]) &
                       (self.bboxes[:, 5] > 0) &
                       (self.bboxes[:, 6] < self.mask.shape[2]))[0]

        # Returns the bounding boxes
        return BBoxes(self.bboxes[idx], self.mask, self.image)

    @staticmethod
    def add_bbox(img, bbox: np.ndarray, save: str = None):
        # bbox is [x_min, x_max, y_min, y_max, z_min, z_max]

        for bbox in bbox:
            z_min, z_max , x_min, x_max, y_min, y_max = bbox[1:].astype(int).tolist()

            # Ensure the bounding box is within the array dimensions
            x_max = min(x_max, img.shape[1] - 1)
            y_max = min(y_max, img.shape[2] - 1)
            z_max = min(z_max, img.shape[0] - 1)

            # Through z_plane
            img[z_min:z_max + 1, x_min, y_min] = 255  # Top left
            img[z_min:z_max + 1, x_min, y_max] = 255  # Top right
            img[z_min:z_max + 1, x_max, y_max] = 255  # Bottom right
            img[z_min:z_max + 1, x_max, y_min] = 255  # Bottom left

            # Top square
            img[z_max, x_min:x_max + 1, y_min] = 255
            img[z_max, x_min:x_max + 1, y_max] = 255
            img[z_max, x_min, y_min:y_max + 1] = 255
            img[z_max, x_max, y_min:y_max + 1] = 255

            # Bottom square
            img[z_min, x_min:x_max + 1, y_min] = 255
            img[z_min, x_min:x_max + 1, y_max] = 255
            img[z_min, x_min, y_min:y_max + 1] = 255
            img[z_min, x_max, y_min:y_max + 1] = 255

        # If save is not None, save the figure
        if save is not None:
            np.save(save, img)

        return img

    @staticmethod
    def _pad_crop(sc: np.ndarray, size: Tuple[int, int, int], v=0) -> np.ndarray:
        """
        Makes an image square to a desire size without changing the ratio

        :param size: Final desired size
        :return: Image of the desired sized, with added padding where needed
        """
        ox = sc.shape[0]
        oy = sc.shape[1]
        oz = sc.shape[2]

        # Get the difference between the current size and the desired size
        dif_x = size[0] - sc.shape[0]
        dif_y = size[1] - sc.shape[1]
        dif_z = size[2] - sc.shape[2]

        # Getting the difference for each size of the image
        dif_x1 = dif_x // 2
        dif_x2 = dif_x // 2 + dif_x % 2
        dif_y1 = dif_y // 2
        dif_y2 = dif_y // 2 + dif_y % 2
        dif_z1 = dif_z // 2
        dif_z2 = dif_z // 2 + dif_z % 2
        diffs = np.array([dif_x1, dif_x2, dif_y1, dif_y2, dif_z1, dif_z2])

        dif_crop = np.where(diffs < 0, -diffs, 0)

        # Get pad differences
        dif_pad = np.where(diffs >= 0, diffs, 0)

        # Remove pixels from image if difference is negative
        sc = sc[0 + dif_crop[0]:ox - dif_crop[1], 0 + dif_crop[2]:oy - dif_crop[3], 0 + dif_crop[4]:oz - dif_crop[5]]

        # Assuming the image is smaller than the desired size
        sc = np.pad(sc, [(dif_pad[0], dif_pad[1]), (dif_pad[2], dif_pad[3]), (dif_pad[4], dif_pad[5])], mode="constant",
                    constant_values=v)

        return sc

    def grab_pixels_from(self,
                         idx: int,
                         source: str = "mask",
                         resize_factor: Union[float, None] = None,
                         size: Tuple[int, int] = None,
                         rescale_intensity: bool = False
                         ) -> np.ndarray:
        """
        Grabs the pixels associated to a single bounding box. The pixels can be grabbed from the mask or the image.

        :param idx: Index of the bounding box to grab.
        :param source: Whether to return the mask or the image associated with the bounding box [default="mask"].
        :param resize_factor: Desired ratio of the bounding box, takes the maximum of the width and height and resizes
        the image to the desired proportion (compared to the size value) while keeping the original aspect ratio.
        :param size: Desired size of the bounding box, final size of the image.
        :param rescale_intensity: Whether to rescale the intensity of the image or not [default=False].
        :return: Mask/image associated with the bounding box.
        """
        # Check if image is not None
        if self.image is None and source == "image":
            self.image = io.imread(self.img_path)

        assert isinstance(self.image, np.ndarray), "Image must be array"

        # Get the single cell image from either mask or image, if neither is selected raise an error
        if source == "mask":
            sc = self.mask[self.bboxes[idx][1]:self.bboxes[idx][2], self.bboxes[idx][3]:self.bboxes[idx][4], self.bboxes[idx][5]:self.bboxes[idx][6]]
        elif source == "image":
            sc = self.image[self.bboxes[idx][1]:self.bboxes[idx][2], self.bboxes[idx][3]:self.bboxes[idx][4], self.bboxes[idx][5]:self.bboxes[idx][6]]
        else:
            raise NotImplementedError("Invalid parameter, please select from 'mask' or 'image'.")

        # Resize the image
        if resize_factor is not None:
            sc = transform.rescale(sc, resize_factor, anti_aliasing=True)

        # Pad and crop the image
        if size is not None:
            sc = self._pad_crop(sc, size)

        # Rescale the intensity
        if rescale_intensity:
            sc = exposure.rescale_intensity(sc, out_range=(0, 255)).astype(np.uint8)

        return sc

    # Saving
    def save_overlapping_pairs(self,
                               output_file: Union[str, Path]) -> None:
        """
        Saves the IoU matrix of all the bounding boxes in the object into a csv file.
        It saves two files: one with the pairs and one with the values.
        :param output_file: Path to the output file(s).
        :return: None
        """
        pairs, values = self.get_overlapping_pairs()

        # Save the IoU matrix to a csv file together with the values
        np.savetxt(output_file, pairs, delimiter=",", fmt="%d")
        np.savetxt(f"{output_file.parent}/{output_file.stem}_values.csv", values, delimiter=",", fmt="%.4f")

    def save_iou_matrix(self,
                        output_file: str) -> None:
        """
        Saves the IoU matrix of all the bounding boxes in the object into a csv file.
        :param output_file: Path to the output file.
        :return: None
        """
        # Save the IoU matrix to a csv file
        np.savetxt(output_file, self.iou_matrix, delimiter=",", fmt="%.2f")

    def save_csv(self,
                 output_file: str) -> None:
        """
        Saves the bounding boxes in the object to a csv file.
        :param output_file: Path to the output file.
        :return:
        """
        # Save the bounding boxes to a csv file
        np.savetxt(output_file, self.bboxes, delimiter=",", fmt="%d")
