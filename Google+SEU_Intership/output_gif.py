import os
from PIL import Image


def create_gif(image_folder, output_gif, duration=25):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

    frames = [Image.open(os.path.join(image_folder, image)) for image in images]

    frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)


image_folder = "./data_KM"
output_gif = "./KM.gif"
create_gif(image_folder, output_gif)
