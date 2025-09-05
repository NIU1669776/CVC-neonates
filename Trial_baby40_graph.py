import os, cv2
import numpy as np
from temp_key_extraction import get_keypoint_temperature
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CustomJS
from bokeh.layouts import column
from bokeh.models.widgets import Button

import matplotlib.pyplot as plt

def process_folders(folders):
    """
    Process all the folders and subfolders to analyze thermal and original images.

    Args:
        folders (list): List of folder paths to process.
    """
    results = []

    for folder in folders:
        for root, _, files in os.walk(folder):
            print(f"Processing folder: {root}")
            print(f"Number of files found: {len(files)}")
            # Filter thermal and original images
            thermals = sorted([f for f in files if not f.endswith(".VIS.jpeg") and f.endswith(".jpeg")])
            originals = sorted([f for f in files if f.endswith(".VIS.jpeg")])
            print(f"Thermal images found: {len(thermals)}")
            print(f"Original images found: {len(originals)}")
            # Match thermal images with their corresponding original images
            for thermal in thermals:
                original_name = thermal.replace(".jpeg", ".VIS.jpeg")
                if original_name in originals:
                    thermal_path = os.path.join(root, thermal)
                    original_path = os.path.join(root, original_name)

                    # Open images
                    thermal_img = cv2.imread(thermal_path)
                    original_img = cv2.imread(original_path)
                    
                    # Call the function and store the result
                    print(f"Processing pair: {thermal_path} and {original_path}")
                    result,_ = get_keypoint_temperature(thermal_img, original_img)
                    print(f"Result: {result}")
                    results.append((thermal_path, result))

    # Sort results by the thermal image path (timeline order)
    results.sort(key=lambda x: x[0])
    print(len(results))
    print([i[0] for i in results])

    # Plot the results
    plot_results(results)

def plot_results(results):
    """
    Plot the results based on the timeline for each keypoint using interactive Bokeh plots.

    Args:
        results (list): List of tuples containing image paths and their results.
    """
    # Extract all keypoint names from the first result
    for res in results:
        if res[1] is not None:
            keypoints = res[1].keys()
            break
    timestamps = list(range(len(results)))

    # Prepare the data for Bokeh
    data = {'timestamps': timestamps}
    for keypoint in keypoints:
        data[keypoint] = [res[1][keypoint] if res[1] is not None else None for res in results]

    source = ColumnDataSource(data=data)

    # Create the figure
    p = figure(title="Temperature Keypoints Over Time",
               x_axis_label="Timeline (Image Index)",
               y_axis_label="Temperature",
               width=800,
               height=400,
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add a line for each keypoint
    renderers = {}
    for keypoint in keypoints:
        line = p.line('timestamps', keypoint, source=source, line_width=2, legend_label=keypoint)
        renderers[keypoint] = line

    # Add hover tool
    hover = HoverTool(tooltips=[("Keypoint", "$name"), ("Temperature", "@$name"), ("Index", "@timestamps")])
    p.add_tools(hover)

    # Add buttons to toggle visibility
    buttons = []
    for keypoint in keypoints:
        button = Button(label=f"Toggle {keypoint}", width=150)
        button.js_on_click(CustomJS(args=dict(renderer=renderers[keypoint]), code="""
            renderer.visible = !renderer.visible;
        """))
        buttons.append(button)

    # Layout the plot and buttons
    layout = column(p, *buttons)

    # Show the plot
    output_notebook()
    show(layout)

# Example usage
folders = [
    "images/40/25.10.24 (2)/"]
print("Processing folders:", folders)

process_folders(folders)

#"images/40/24.10.24-25.10.24/","images/40/25.10.24/",