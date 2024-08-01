"""
Package Configurations
python                    3.11.3
simpy                     4.0.1
"""
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
# If you also want to get the image bytes as a variable, you can use BytesIO
from io import BytesIO

simmode = ''

# create a column with the color for each department
def color(row):
    c_dict = {'Part0': '#0000ff', 'Part1': '#ffa500', 'Part2': '#006400',
              'Part3': '#ff0000', 'Part4': '#cdc0b0', 'Part5': '#66cdaa',
              'Part6': '#1abc9c','Part7': '#a52a2a','Part8': '#5bc0de',
              'Part9': '#fc8c84'}
    return c_dict[row['Job'][0:5]]


def Gantt(result, num, config):

    df = result.iloc[0:num].copy()

    df['color'] = df.apply(color, axis=1)

    fig, ax = plt.subplots(1, figsize=(16*0.8, 9*0.8))
    ax.barh(df.Machine, df.Delta, left=df.Start, color=df.color, edgecolor='black')
    ##### LEGENDS #####
    c_dict = {'Part0': '#0000ff', 'Part1': '#ffa500', 'Part2': '#006400',
              'Part3': '#ff0000', 'Part4': '#cdc0b0', 'Part5': '#66cdaa',
              'Part6': '#1abc9c','Part7': '#a52a2a','Part8': '#5bc0de',
              'Part9': '#fc8c84'}
    legend_elements = [Patch(facecolor=c_dict[i], label=i) for i in c_dict]
    plt.legend(handles=legend_elements)
    plt.title(config.gantt_title, size=24)

    ##### TICKS #####
    if config.show_gantt:
        plt.show()

    # Save the figure as an image file
    if config.save_gantt:
        fig.savefig(config.filename['gantt'], format='png')

    # Create a BytesIO object
    image_bytes_io = BytesIO()

    # Save the figure to the BytesIO object
    fig.savefig(image_bytes_io, format='png')  # This is different from saving file as .png

    # Get the image bytes
    image_bytes = image_bytes_io.getvalue()

    return image_bytes

