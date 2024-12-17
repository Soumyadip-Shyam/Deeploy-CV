from PIL import Image
import numpy as np
import requests
from io import BytesIO

def flag(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((100, 100))
    img_array = np.array(img)

    def is_red(pixel):
        r, g, b = pixel
        return r > 150 and g < 100 and b < 100

    def is_white(pixel):
        r, g, b = pixel
        return r > 200 and g > 200 and b > 200

    height, width, _ = img_array.shape
    top_half = img_array[:height // 2]
    bottom_half = img_array[height // 2:]

    top_red_count = np.sum([is_red(pixel) for row in top_half for pixel in row])
    top_white_count = np.sum([is_white(pixel) for row in top_half for pixel in row])
    bottom_red_count = np.sum([is_red(pixel) for row in bottom_half for pixel in row])
    bottom_white_count = np.sum([is_white(pixel) for row in bottom_half for pixel in row])

    if top_red_count > top_white_count and bottom_white_count > bottom_red_count:
        return "Indonesia Flag"
    elif top_white_count > top_red_count and bottom_red_count > bottom_white_count:
        return "Poland Flag"
    else:
        return "Not Recognized as Indonesia or Poland Flag"

if __name__ == "__main__":
    image_path = input("Enter the path to the image or URL: ")
    result = flag(image_path)
    print("Result:", result)

