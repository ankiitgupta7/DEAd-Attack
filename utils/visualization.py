# /utils/visualization.py
import matplotlib.pyplot as plt

def visualize_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
