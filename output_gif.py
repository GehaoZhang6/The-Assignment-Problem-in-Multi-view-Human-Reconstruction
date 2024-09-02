import os
from PIL import Image


def create_gif(image_folder, output_gif, duration=25):
    # 获取文件夹中所有图片的文件名，并按照文件名排序
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

    # 打开图片并将其添加到列表中
    frames = [Image.open(os.path.join(image_folder, image)) for image in images]

    # 将图片保存为GIF
    frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)


# 使用示例
image_folder = "./data_KM"  # 你的图片文件夹路径
output_gif = "./KM.gif"  # 输出GIF文件路径
create_gif(image_folder, output_gif)
