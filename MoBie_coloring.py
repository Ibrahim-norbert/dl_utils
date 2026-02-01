import os

import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class GlasbeyARGBLut:
    MINIMUM_RGB_DIFFERENCE = 50
    MINIMUM_SUM_BRIGHTNESS = 150

    def __init__(self, alpha=255):
        self.alpha = alpha
        self.indices = self._argb_glasbey_indices()
        self.num_colors = len(self.indices)
        self.name = "GLASBEY"

    def get_argb(self, x):
        try:
            index = int(x * (self.num_colors - 1))
            return self.indices[index]
        except IndexError:
            raise RuntimeError("Index out of bounds")

    def get_argb_by_index(self, i):
        index = i % self.num_colors
        return self.indices[index]

    @staticmethod
    def rgba(r, g, b, a):
        return (a << 24) | (r << 16) | (g << 8) | b

    def rgba_tuple_by_index(self, i: int):
        if i > 0:
            argb = self.get_argb_by_index(i)
            a = (argb >> 24) & 0xFF
            r = (argb >> 16) & 0xFF
            g = (argb >> 8) & 0xFF
            b = argb & 0xFF
        else:
            r, g, b, a = 128, 128, 128, 1
        return (r, g, b, a)

    def rgba_plt_by_index(self, i: int):
        r, g, b, a = self.rgba_tuple_by_index(i)
        # Normalize the components to [0, 1]
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    def rgba_mobie_by_index(self, i: int):
        r, g, b, a = self.rgba_tuple_by_index(i)
        return f"{r}-{g}-{b}-{a}"

    def _argb_glasbey_indices(self):
        r = [0, 0, 255, 0, 0, 255, 0, 255, 0, 154, 0, 120, 31, 255, 177, 241, 254, 221, 32, 114, 118, 2, 200, 136, 255, 133, 161, 20, 0, 220, 147, 0, 0, 57, 238, 0, 171, 161, 164, 255, 71, 212, 251, 171, 117, 166, 0, 165, 98, 0, 0, 86, 159, 66, 255, 0, 252, 159, 167, 74, 0, 145, 207, 195, 253, 66, 106, 181, 132, 96, 255, 102, 254, 228, 17, 210, 91, 32, 180, 226, 0, 93, 166, 97, 98, 126, 0, 255, 7, 180, 148, 204, 55, 0, 150, 39, 206, 150, 180, 110, 147, 199, 115, 15, 172, 182, 216, 87, 216, 0, 243, 216, 1, 52, 255, 87, 198, 255, 123, 120, 162, 105, 198, 121, 0, 231, 217, 255, 209, 36, 87, 211, 203, 62, 0, 112, 209, 0, 105, 255, 233, 191, 69, 171, 14, 0, 118, 255, 94, 238, 159, 80, 189, 0, 88, 71, 1, 99, 2, 139, 171, 141, 85, 150, 0, 255, 222, 107, 30, 173, 255, 0, 138, 111, 225, 255, 229, 114, 111, 134, 99, 105, 200, 209, 198, 79, 174, 170, 199, 255, 146, 102, 111, 92, 172, 210, 199, 255, 250, 49, 254, 254, 68, 201, 199, 68, 147, 22, 8, 116, 104, 64, 164, 207, 118, 83, 0, 43, 160, 176, 29, 122, 214, 160, 106, 153, 192, 125, 149, 213, 22, 166, 109, 86, 255, 255, 255, 202, 67, 234, 191, 38, 85, 121, 254, 139, 141, 0, 63, 255, 17, 154, 149, 126, 58, 189]
        g = [0, 0, 0, 255, 0, 0, 83, 211, 159, 77, 255, 63, 150, 172, 204, 8, 143, 0, 26, 0, 108, 173, 255, 108, 183, 133, 3, 249, 71, 94, 212, 76, 66, 167, 112, 0, 245, 146, 255, 206, 0, 173, 118, 188, 0, 0, 115, 93, 132, 121, 255, 53, 0, 45, 242, 93, 255, 191, 84, 39, 16, 78, 149, 187, 68, 78, 1, 131, 233, 217, 111, 75, 100, 3, 199, 129, 118, 59, 84, 8, 1, 132, 250, 123, 0, 190, 60, 253, 197, 167, 186, 187, 0, 40, 122, 136, 130, 164, 32, 86, 0, 48, 102, 187, 164, 117, 220, 141, 85, 196, 165, 255, 24, 66, 154, 95, 241, 95, 172, 100, 133, 255, 82, 26, 238, 207, 128, 211, 255, 0, 163, 231, 111, 24, 117, 176, 24, 30, 200, 203, 194, 129, 42, 76, 117, 30, 73, 169, 55, 230, 54, 0, 144, 109, 223, 80, 93, 48, 206, 83, 0, 42, 83, 255, 152, 138, 69, 109, 0, 76, 134, 35, 205, 202, 75, 176, 232, 16, 82, 137, 38, 38, 110, 164, 210, 103, 165, 45, 81, 89, 102, 134, 152, 255, 137, 34, 207, 185, 148, 34, 81, 141, 54, 162, 232, 152, 172, 75, 84, 45, 60, 41, 113, 0, 1, 0, 82, 92, 217, 26, 3, 58, 209, 100, 157, 219, 56, 255, 0, 162, 131, 249, 105, 188, 109, 3, 0, 0, 109, 170, 165, 44, 185, 182, 236, 165, 254, 60, 17, 221, 26, 66, 157, 130, 6, 117]
        b = [0, 255, 0, 0, 51, 182, 0, 0, 255, 66, 190, 193, 152, 253, 113, 92, 66, 255, 1, 85, 149, 36, 0, 0, 159, 103, 0, 255, 158, 147, 255, 255, 80, 106, 254, 100, 204, 255, 115, 113, 21, 197, 111, 0, 215, 154, 254, 174, 2, 168, 131, 0, 63, 66, 187, 67, 124, 186, 19, 108, 166, 109, 0, 255, 64, 32, 0, 84, 147, 0, 211, 63, 0, 127, 174, 139, 124, 106, 255, 210, 20, 68, 255, 201, 122, 58, 183, 0, 226, 57, 138, 160, 49, 1, 129, 38, 180, 196, 128, 180, 185, 61, 255, 253, 100, 250, 254, 113, 34, 103, 105, 182, 219, 54, 0, 1, 79, 133, 240, 49, 204, 220, 100, 64, 70, 69, 233, 209, 141, 3, 193, 201, 79, 0, 223, 88, 0, 107, 197, 255, 137, 46, 145, 194, 61, 25, 127, 200, 217, 138, 33, 148, 128, 126, 96, 103, 159, 60, 148, 37, 255, 135, 148, 0, 123, 203, 200, 230, 68, 138, 161, 60, 0, 157, 253, 77, 57, 255, 101, 48, 80, 32, 0, 255, 86, 77, 166, 101, 175, 172, 78, 184, 255, 159, 178, 98, 147, 30, 141, 78, 97, 100, 23, 84, 240, 0, 58, 28, 121, 0, 255, 38, 215, 155, 35, 88, 232, 87, 146, 229, 36, 159, 207, 105, 160, 113, 207, 89, 34, 223, 204, 69, 97, 78, 81, 248, 73, 35, 18, 173, 0, 51, 2, 158, 212, 89, 193, 43, 40, 246, 146, 84, 238, 72, 101, 101]

        indices = []

        indices.append(self.rgba(0, 0, 0, 0))

        for i in range(1, len(r)):
            if (abs(g[i] - r[i]) < self.MINIMUM_RGB_DIFFERENCE and
                    abs(g[i] - b[i]) < self.MINIMUM_RGB_DIFFERENCE and
                    abs(r[i] - b[i]) < self.MINIMUM_RGB_DIFFERENCE):
                continue  # too grayish

            if (r[i] + g[i] + b[i]) < self.MINIMUM_SUM_BRIGHTNESS:
                continue  # too dark

            indices.append(self.rgba(r[i], g[i], b[i], self.alpha))

        return indices


class SeismicARGBLut(GlasbeyARGBLut):
    def __init__(self, alpha=255, num_colors=256):
        super().__init__(alpha)
        self.alpha = alpha
        self.num_colors = num_colors
        self.name = "SEISMIC"
        self.indices = self._generate_seismic_indices()

    def _generate_seismic_indices(self):
        cmap = plt.cm.seismic  # Get the seismic colormap
        colors = cmap(np.linspace(0, 1, self.num_colors))  # Sample colors
        indices = [
            self.rgba(int(r * 255), int(g * 255), int(b * 255), self.alpha)
            for r, g, b, a in colors  # a is not used; alpha is manually set
        ]
        return indices

    def get_argb(self, x):
        index = int(x * (self.num_colors - 1))
        return self.indices[index]

    def rgba_tuple_by_normalized(self, x: float):
        """
        Get the RGBA tuple corresponding to a normalized value.

        Parameters:
        - x: Normalized value (0 to 1).

        Returns:
        - A tuple of (r, g, b, a).
        """
        if not np.isnan(x):
            argb = self.get_argb(x)
            a = (argb >> 24) & 0xFF
            r = (argb >> 16) & 0xFF
            g = (argb >> 8) & 0xFF
            b = argb & 0xFF
        else:
            r, g, b, a = 128, 128, 128, 1

        return (r, g, b, a)

    def rgba_plt_by_normalized(self, x: float):
        r, g, b, a = self.rgba_tuple_by_normalized(x)
        # Normalize the components to [0, 1]
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    def rgba_mobie_by_normalized(self, x):
        """
        Get the RGBA string in MoBIE format for a normalized value.

        Parameters:
        - x: Normalized value (0 to 1).

        Returns:
        - A string in the format "r-g-b-a".
        """
        r, g, b, a = self.rgba_tuple_by_normalized(x)
        return f"{r}-{g}-{b}-{a}"

    def rgba_plotly_by_normalized(self, x):
        r, g, b, a = self.rgba_tuple_by_normalized(x)
        return f"rgb({r},{g},{b})"

    @staticmethod
    def rgba(r, g, b, a):
        return (a << 24) | (r << 16) | (g << 8) | b

    def get_colorgradient_plot(self, orientation="vertical", save_dir=None):
        """
        Plot a specified colormap with ticks for minimum and maximum values.

        Parameters:
        - cmap_name (str): Name of the colormap to display (e.g., 'viridis', 'plasma', 'cividis').
        - orientation (str): Orientation of the color gradient ('horizontal' or 'vertical').
        - min_val (float): Minimum value to display as tick on the gradient.
        - max_val (float): Maximum value to display as tick on the gradient.
        """
        # Create a gradient image for displaying the colormap
        gradient = np.linspace(0, 1, self.num_colors)
        colors = gradient
        gradient = np.vstack([gradient, gradient]) if orientation == "horizontal" else np.hstack([gradient[:, None]] * 2)

        # Plot the colormap
        fig, ax = plt.subplots(figsize=(6, 1) if orientation == "horizontal" else (1, 6), dpi= 400)
        colors = [np.array(self.rgba_tuple_by_normalized(x)) / 255 for x in colors]

        custom_cmap = mcolors.ListedColormap(colors)

        ax.imshow(gradient, aspect="auto", cmap=custom_cmap)

        median = self.num_colors // 2
        min_val = 0
        max_val = self.num_colors
        # Set axis properties based on orientation
        if orientation == "horizontal":
            ax.set_xticks([min_val, median, max_val])  # Ticks at both ends
            ax.set_xticklabels([str(min_val), str(median/self.num_colors), str(max_val/self.num_colors)],
                               fontsize=20, color="black", rotation=270.)
            ax.set_yticks([])
        else:
            ax.set_yticks([min_val, median, max_val])  # Ticks at both ends
            ax.set_yticklabels([str(min_val), str(median/self.num_colors), str(max_val/self.num_colors)],
                               fontsize=20, color="black")
            ax.set_xticks([])
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir,f"{self.name}_colorbar.svg"))



class ViridisARGBLut(SeismicARGBLut):

    def __init__(self, alpha=255, num_colors=256):
        super().__init__(alpha, num_colors)
        """
        Initialize the colormap with a given alpha transparency.

        Parameters:
        - alpha (int): Alpha value (0-255) for transparency.
        """
        self.alpha = alpha
        self.indices = self.argb_indices()
        self.num_colors = len(self.indices)
        self.name = "Viridis".upper()

    def get_argb(self, x):
        """
        Get the ARGB color corresponding to a normalized value (0 to 1).

        Parameters:
        - x (float): Normalized value between 0 and 1.

        Returns:
        - int: ARGB color as a packed integer.
        """
        index = min(int(np.ceil(x * self.num_colors)), self.num_colors - 1)
        return self.indices[index]

    def argb_indices(self):
        """
        Generate an ARGB lookup table from the Viridis colormap.

        Returns:
        - np.ndarray: Array of ARGB color values.
        """
        # Get 255 colors from the Viridis colormap
        viridis_colors = cm.get_cmap("viridis", self.num_colors)

        # Convert colors to 8-bit RGB format and apply alpha
        argb_indices = np.array([
            self.rgba_to_argb(
                int(255 * r), int(255 * g), int(255 * b), self.alpha
            )
            for r, g, b, _ in viridis_colors(np.linspace(0, 1, self.num_colors))
        ], dtype=np.uint32)

        return argb_indices

    @staticmethod
    def rgba_to_argb(r, g, b, a):
        """
        Convert RGBA values to a single ARGB integer.

        Parameters:
        - r, g, b, a (int): Red, Green, Blue, Alpha values (0-255).

        Returns:
        - int: ARGB integer representation.
        """
        return (a << 24) | (r << 16) | (g << 8) | b  # Packing into ARGB format
