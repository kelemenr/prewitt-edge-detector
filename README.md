# Edge Detection with Prewitt Filter

This Python script performs edge detection on an input grayscale image using the Prewitt filter. It emphasizes edges in the image and applies non-maximum suppression to produce a clear representation of edges.

## Features

- Edge detection using the Prewitt filter.
- Non-maximum suppression for thinning edges.
- Display of original image, Prewitt filtered image, and non-maximum suppressed image.

## Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- Scikit-image (`skimage`)
- SciPy

## Usage

1. Clone the repository or download the script.
2. Run the script by providing the path to an image as a command-line argument:

   ```shell
   python edge_detection.py <image_path>
   ```
Replace <image_path> with the path to your grayscale image file.

## Example

Here's an example of how to use the script:
```shell
   python edge_detection.py <image_path>
```
The script will display the original image, the Prewitt filtered image, and the non-maximum suppressed image side by side.

## Output

The processed images (Prewitt filtered and non-maximum suppressed) are saved in the output/ directory with the filenames prewitt_<input_image> and non_max_<input_image>.

