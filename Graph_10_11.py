import os, cv2
import numpy as np
from datetime import datetime
from temp_key_extraction import get_keypoint_temperature
from thermal import limit_finder
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource, CustomJS, Range1d
from bokeh.layouts import row, column
from bokeh.models.widgets import Button
from bokeh.palettes import Category10
import mediapipe as mp
from bokeh.events import Reset # Import Reset

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
                    TOP,BOT = limit_finder(thermal_img)
                    error = min((TOP-BOT)*0.02,2)
                    result, _ = get_keypoint_temperature(thermal_img, original_img, pose)
                    print(f"Result: {result}")
                    
                    # Store the error along with the result
                    results.append((thermal_path, result, error)) 

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

    # Extract keypoints from first valid result
    keypoints = []
    for _, res, _ in results: # Unpack 3 items
        if isinstance(res, dict) and res:
            keypoints = list(res.keys())
            break

    if not keypoints:
        print("No keypoints found to plot.")
        return

    # Extract real timestamps and a single error column for hover
    timestamps = []
    errors = [] 
    for path, _, err_val in results: 
        timestamps.append(_extract_datetime_from_filename(path))
        errors.append(_safe_val(err_val)) 
    
    data = {'timestamps': timestamps, 'error': errors} 

    # --- MODIFICATION: Prepare data with helper column ---
    for keypoint in keypoints:
        col_val = []
        col_upper = []
        col_lower = []
        
        for (path, res, err_val), err_safe in zip(results, errors): # Zip with pre-safed errors
            val = _safe_val(res.get(keypoint) if isinstance(res, dict) else None)
            
            col_val.append(val)
            
            if np.isnan(val) or np.isnan(err_safe):
                col_upper.append(np.nan), col_lower.append(np.nan)
            else:
                col_upper.append(val + err_safe), col_lower.append(val - err_safe)

        data[keypoint] = col_val
        data[f"{keypoint}_upper"] = col_upper
        data[f"{keypoint}_lower"] = col_lower
        
        # --- THIS IS THE KEY ---
        # Add a new column that just contains the keypoint name (e.g., "left_wrist")
        # This is what we will use for the legend.
        data[f"{keypoint}_name"] = [keypoint] * len(timestamps)
        # -----------------------
        
    source = ColumnDataSource(data=data)
    # --- END DATA PREP ---

    # --- Main plot (temperature over time) ---
    p = figure(title="Temperature Keypoints Over Time (Interactive)",
               x_axis_type="datetime",
               x_axis_label="Timestamp",
               y_axis_label="Temperature",
               sizing_mode="stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset,save")

    # Fixed colors
    palette = Category10[10] if len(keypoints) <= 10 else (Category10[10] * ((len(keypoints)//10)+1))
    lines = {}
    circles = {}
    areas = {} 

    for i, keypoint in enumerate(keypoints):
        color = palette[i % len(palette)]
        
        # --- MODIFICATION: Use legend_field ---
        # We point all three glyphs to the *same* helper column.
        # Bokeh automatically groups them under one legend entry.
        
        legend_col_name = f"{keypoint}_name"

        areas[keypoint] = p.varea(
            x='timestamps',
            y1=f"{keypoint}_lower",
            y2=f"{keypoint}_upper",
            source=source,
            fill_color=color, 
            fill_alpha=0.5,
            name=keypoint, 
            visible=True,
            legend_field=legend_col_name  # <-- Use legend_field
        )
        
        lines[keypoint] = p.line(
            'timestamps', keypoint, source=source, line_width=2,
            color=color, 
            name=keypoint, 
            visible=True,
            legend_field=legend_col_name  # <-- Use legend_field (replaces legend_label)
        )
        
        circles[keypoint] = p.circle(
            'timestamps', keypoint, source=source, size=6,
            color=color, alpha=0.9, line_color=None,
            name=keypoint, # This name is used by the HoverTool
            visible=True, 
            legend_field=legend_col_name  # <-- Use legend_field
        )
        # ----------------------------------------

    # Hover on points
    hover = HoverTool(
        tooltips=[
            ("Keypoint", "$name"), # $name reads the 'name' prop from the glyph (which we set to keypoint)
            ("Temperature", f"$y (Error: +/- @error{{0.00}})"),
            ("Timestamp", "@timestamps{%F %T}"),
        ],
        formatters={"@timestamps": "datetime"},
        mode="mouse",
        renderers=list(circles.values()), # Only hover on the circles
    )
    p.add_tools(hover)

    # ✅ This now works correctly!
    p.legend.click_policy = "hide" # "hide" or "mute" both work
    p.legend.title = "Click to toggle keypoints"
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "10pt"

    # Reset event: restore all lines (this JS is still correct)
    p.js_on_event(Reset, CustomJS(args=dict(lines=lines, circles=circles, areas=areas), code="""
        for (const k in lines) {
            lines[k].visible = true;
            circles[k].visible = true;
            if (areas[k]) {
                areas[k].visible = true;
            }
        }
    """))

    # --- Image subplot ---
    urls, xs, ys = [], [], []
    for i, (path, _, _) in enumerate(results): # Unpack 3
        if i % 5 == 0:  # every 5th point
            original_path = path.replace(".jpeg", ".VIS.jpeg")
            if os.path.exists(original_path):
                urls.append(original_path)
                xs.append(timestamps[i])
                ys.append(0.5)  # vertical position (center in [0,1])

    if urls:
        img_source = ColumnDataSource(data=dict(url=urls, x=xs, y=ys))
        diffs = np.diff([dt.timestamp() for dt in timestamps if dt is not None])
        median_step = np.median(diffs) * 1000 if len(diffs) else 60000  # fallback 1 min
        img_plot = figure(x_axis_type="datetime", x_range=p.x_range,
                          y_range=Range1d(0, 1), height=180,
                          sizing_mode="stretch_width", toolbar_location=None)
        fixed_w = int(median_step * 4)
        fixed_h = 0.8
        img_plot.image_url(url="url", x="x", y="y", w=fixed_w, h=fixed_h,
                           anchor="center", source=img_source)
        img_plot.yaxis.visible = False
        img_plot.xaxis.visible = False
        img_plot.grid.visible = False
        layout = column(p, img_plot, sizing_mode="stretch_both")
    else:
        layout = column(p, sizing_mode="stretch_both")

    output_file("keypoints_plot.html")
    show(layout)


# Example usage
if __name__ == "__main__":
    # Make sure to replace this with your actual folder path
    folders = ["images/NEONATE No 85"] 
    print("Processing folders:", folders)
    process_folders(folders)