import os, cv2
import numpy as np
from datetime import datetime
from temp_key_extraction import get_keypoint_temperature
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource, CustomJS, Range1d
from bokeh.layouts import row, column
from bokeh.models.widgets import Button
from bokeh.palettes import Category10
import mediapipe as mp

mp_pose = mp.solutions.pose

def process_folders(folders):
    """
    Process all the folders and subfolders to analyze thermal and original images.
    """
    pose = None
    results = []

    if len(folders) > 1:
        pose = mp_pose.Pose(static_image_mode=False)

    for folder in folders:
        for root, _, files in os.walk(folder):
            print(f"Processing folder: {root}")
            print(f"Number of files found: {len(files)}")

            if len(files)>1 and pose is None:
                pose = mp_pose.Pose(static_image_mode=False)
            
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
                    result, _ = get_keypoint_temperature(thermal_img, original_img, pose)
                    print(f"Result: {result}")
                    results.append((thermal_path, result))

    # Sort results by the thermal image path (timeline order)
    results.sort(key=lambda x: x[0])
    print(len(results))
    print([i[0] for i in results])

    # Plot the results
    if results:
        plot_results(results)
    else:
        print("No results to plot.")

def _safe_val(v):
    """Return a float or np.nan for invalid/missing values."""
    if v is None:
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan

def _extract_datetime_from_filename(path):
    """
    Extract datetime from filenames like HM20241025142419.jpeg
    """
    fname = os.path.basename(path)
    try:
        idx = fname.index("HM")
        timestamp_str = fname[idx+2:idx+16]  # 14 chars after HM
        dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        return dt
    except Exception as e:
        print(f"Warning: could not parse datetime from {fname} ({e})")
        return None
    
    
def plot_results(results):
    """
    Plot the results based on the timeline for each keypoint using interactive Bokeh plots,
    with the corresponding original images displayed every 5 points, centered on their timestamps.
    """
    from bokeh.plotting import figure, show, output_file
    from bokeh.models import HoverTool, ColumnDataSource, CustomJS, Range1d
    from bokeh.layouts import column
    from bokeh.palettes import Category10
    from bokeh.events import Reset
    import numpy as np
    import os

    # Extract keypoints from first valid result
    keypoints = []
    for _, res in results:
        if isinstance(res, dict) and res:
            keypoints = list(res.keys())
            break

    if not keypoints:
        print("No keypoints found to plot.")
        return

    # Extract real timestamps from filenames
    timestamps = []
    for path, _ in results:
        dt = _extract_datetime_from_filename(path)
        timestamps.append(dt)

    # Prepare data (coerce missing/invalid to NaN)
    data = {'timestamps': timestamps}
    for keypoint in keypoints:
        col = []
        for _, res in results:
            if isinstance(res, dict) and (keypoint in res):
                col.append(_safe_val(res.get(keypoint)))
            else:
                col.append(np.nan)
        data[keypoint] = col

    source = ColumnDataSource(data=data)

    # --- Main plot (temperature over time) ---
    p = figure(title="Temperature Keypoints Over Time",
               x_axis_type="datetime",
               x_axis_label="Timestamp",
               y_axis_label="Temperature",
               sizing_mode="stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Fixed colors
    palette = Category10[10] if len(keypoints) <= 10 else (Category10[10] * ((len(keypoints)//10)+1))
    lines = {}
    circles = {}

    for i, keypoint in enumerate(keypoints):
        color = palette[i % len(palette)]
        # Line (broken where NaN)
        lines[keypoint] = p.line('timestamps', keypoint,
                                 source=source, line_width=2,
                                 color=color, legend_label=keypoint, name=keypoint,
                                 visible=True)
        # Circle markers only at valid points
        circles[keypoint] = p.circle('timestamps', keypoint,
                                     source=source, size=6,
                                     color=color, alpha=0.9, line_color=None,
                                     name=keypoint, visible=True)

    # Hover only works on circles
    hover = HoverTool(
        tooltips=[
            ("Keypoint", "$name"),
            ("Temperature", "$y"),
            ("Timestamp", "@timestamps{%F %T}"),
        ],
        formatters={"@timestamps": "datetime"},
        mode="mouse",
        renderers=list(circles.values()),
    )
    p.add_tools(hover)

    # Reset event: restore all lines
    p.js_on_event(Reset, CustomJS(args=dict(lines=lines, circles=circles), code="""
        for (const k in lines) {
            lines[k].visible = true;
            circles[k].visible = true;
        }
    """))

    # --- Image subplot ---
    urls, xs, ys = [], [], []
    for i, (path, _) in enumerate(results):
        if i % 5 == 0:  # every 5th point
            original_path = path.replace(".jpeg", ".VIS.jpeg")
            if os.path.exists(original_path):
                urls.append(original_path)
                xs.append(timestamps[i])
                ys.append(0.5)  # vertical position (center in [0,1])

    if urls:
        img_source = ColumnDataSource(data=dict(url=urls, x=xs, y=ys))

        # Estimate median step (ms) between timestamps for scaling width
        diffs = np.diff([dt.timestamp() for dt in timestamps if dt is not None])
        median_step = np.median(diffs) * 1000 if len(diffs) else 60000  # fallback: 1 minute in ms

        img_plot = figure(x_axis_type="datetime",
                          x_range=p.x_range,
                          y_range=Range1d(0, 1),
                          height=180,
                          sizing_mode="stretch_width",
                          toolbar_location=None)

        # Maintain correct aspect ratio (408x544)
        fixed_w = int(median_step * 4)  # width in ms (data units)
        fixed_h = 0.8  # relative to y_range [0,1]

        img_plot.image_url(url="url",
                           x="x", y="y",
                           w=fixed_w, h=fixed_h,
                           anchor="center",   # <-- center at timestamp
                           source=img_source)

        img_plot.yaxis.visible = False
        img_plot.xaxis.visible = False
        img_plot.grid.visible = False

        layout = column(p, img_plot, sizing_mode="stretch_both")
    else:
        layout = column(p, sizing_mode="stretch_both")

    # Save to HTML and open in browser
    output_file("keypoints_plot.html")
    show(layout)


# Example usage
if __name__ == "__main__":
    folders = ["Trial_folder"]
    print("Processing folders:", folders)
    process_folders(folders)
