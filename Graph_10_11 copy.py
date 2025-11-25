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

    # --- (Data Prep section is unchanged) ---
    keypoints = []
    for _, res, _ in results: 
        if isinstance(res, dict) and res:
            keypoints = list(res.keys())
            break

    if not keypoints:
        print("No keypoints found to plot.")
        return

    timestamps = []
    errors = [] 
    for path, _, err_val in results: 
        timestamps.append(_extract_datetime_from_filename(path))
        errors.append(_safe_val(err_val)) 
    
    data = {'timestamps': timestamps, 'error': errors} 

    for keypoint in keypoints:
        col_val = []
        col_upper = []
        col_lower = []
        
        for (path, res, err_val), err_safe in zip(results, errors):
            val = _safe_val(res.get(keypoint) if isinstance(res, dict) else None)
            col_val.append(val)
            
            if np.isnan(val) or np.isnan(err_safe):
                col_upper.append(np.nan), col_lower.append(np.nan)
            else:
                col_upper.append(val + err_safe), col_lower.append(val - err_safe)

        data[keypoint] = col_val
        data[f"{keypoint}_upper"] = col_upper
        data[f"{keypoint}_lower"] = col_lower
        data[f"{keypoint}_name"] = [keypoint] * len(timestamps)
        
    source = ColumnDataSource(data=data)
    # --- (End Data Prep) ---

    # --- (Main plot and glyphs section is unchanged) ---
    p = figure(title="Temperature Keypoints Over Time (Interactive)",
               x_axis_type="datetime",
               x_axis_label="Timestamp",
               y_axis_label="Temperature",
               sizing_mode="stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset,save")

    palette = Category10[10] if len(keypoints) <= 10 else (Category10[10] * ((len(keypoints)//10)+1))
    lines = {}
    circles = {}
    areas = {} 

    for i, keypoint in enumerate(keypoints):
        color = palette[i % len(palette)]
        legend_col_name = f"{keypoint}_name"

        areas[keypoint] = p.varea(
            x='timestamps', y1=f"{keypoint}_lower", y2=f"{keypoint}_upper",
            source=source, fill_color=color, fill_alpha=0.5, name=keypoint,
            visible=True, legend_field=legend_col_name
        )
        lines[keypoint] = p.line(
            'timestamps', keypoint, source=source, line_width=2, color=color,
            name=keypoint, visible=True, legend_field=legend_col_name
        )
        circles[keypoint] = p.circle(
            'timestamps', keypoint, source=source, size=6, color=color,
            alpha=0.9, line_color=None, name=keypoint, visible=True,
            legend_field=legend_col_name
        )

    hover = HoverTool(
        tooltips=[
            ("Keypoint", "$name"),
            ("Temperature", f"$y (Error: +/- @error{{0.00}})"),
            ("Timestamp", "@timestamps{%F %T}"),
        ],
        formatters={"@timestamps": "datetime"},
        mode="mouse",
        renderers=list(circles.values()),
    )
    p.add_tools(hover)

    p.legend.click_policy = "hide"
    p.legend.title = "Click to toggle keypoints"
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "10pt"
    # --- (End Main plot) ---


    # --- MODIFICATION: Image Subplot ---
    
    # 1. Get ALL image URLs and timestamps
    all_urls, all_xs, all_ys = [], [], []
    for i, (path, _, _) in enumerate(results): 
        original_path = path.replace(".jpeg", ".VIS.jpeg")
        dt = timestamps[i] # Use the already-parsed timestamps
        if os.path.exists(original_path) and dt is not None:
            all_urls.append(original_path)
            all_xs.append(dt)
            all_ys.append(0.5)

    if all_urls:
        # 2. Create the hidden source with all image data
        all_images_source = ColumnDataSource(data=dict(url=all_urls, x=all_xs, y=all_ys))
        
        # 3. Perform initial decimation in Python for the first load
        initial_x, initial_url, initial_w, initial_y = [], [], [], []
        target_images = 10 # Target number of images
        
        timestamps_ms = [dt.timestamp() * 1000 for dt in all_xs]
        if timestamps_ms:
            start_ms = timestamps_ms[0]
            end_ms = timestamps_ms[-1]
            view_duration_ms = end_ms - start_ms
            
            # Calculate width in milliseconds
            new_width_ms = (view_duration_ms / target_images) * 0.9 # 90% width
            
            step = max(1, len(all_xs) // target_images)
            for i in range(0, len(all_xs), step):
                initial_x.append(all_xs[i])
                initial_url.append(all_urls[i])
                initial_w.append(new_width_ms)
                initial_y.append(0.5)
        
        # 4. Create the visible source, pre-populated with the initial set
        img_source = ColumnDataSource(data=dict(url=initial_url, x=initial_x, y=initial_y, w=initial_w))

        # 5. Create the image plot
        img_plot = figure(x_axis_type="datetime", x_range=p.x_range,
                          y_range=Range1d(0, 1), height=180,
                          sizing_mode="stretch_width", toolbar_location=None)
        
        # 6. Change image_url to use the dynamic 'w' column
        fixed_h = 0.8 # Height can remain fixed
        img_plot.image_url(url="url", x="x", y="y", w='w', h=fixed_h,
                           anchor="center", source=img_source)
        
        img_plot.yaxis.visible = False
        img_plot.xaxis.visible = False
        img_plot.grid.visible = False
        
        # 7. Define the JavaScript callback for decimation
        JS_CODE_DECIMATE_IMAGES = """
            const all_data = all_images_source.data;
            const img_data = img_source.data;
            const all_x = all_data['x'];
            const all_url = all_data['url'];
            
            // Get current view range from the main plot's x_range
            const start = x_range.start;
            const end = x_range.end;
            const view_duration = end - start; // Duration in milliseconds

            const TARGET_IMAGES = 10; // Target number of images in view
            // Calculate new width (90% of the slot for one image)
            const new_width = (view_duration / TARGET_IMAGES) * 0.9;

            const new_x = [], new_url = [], new_w = [], new_y = [];

            // Find indices of images within the view (with a buffer)
            const indices = [];
            for (let i = 0; i < all_x.length; i++) {
                if (all_x[i] >= start - new_width/2 && all_x[i] <= end + new_width/2) {
                    indices.push(i);
                }
            }

            // Decimate the visible images
            const step = Math.max(1, Math.floor(indices.length / TARGET_IMAGES));
            for (let i = 0; i < indices.length; i += step) {
                let idx = indices[i];
                new_x.push(all_x[idx]);
                new_url.push(all_url[idx]);
                new_w.push(new_width);
                new_y.push(0.5); 
            }
            
            // Update the visible source's data, which triggers the plot to re-draw
            img_source.data = {'x': new_x, 'url': new_url, 'w': new_w, 'y': new_y};
        """
        
        # 8. Attach this callback to Pan/Zoom events (with throttling)
        pan_zoom_callback = CustomJS(
            args=dict(all_images_source=all_images_source, img_source=img_source, x_range=p.x_range),
            code=JS_CODE_DECIMATE_IMAGES
        )
        # Throttle set to 200ms, so it doesn't fire on every tiny mouse movement
        p.x_range.js_on_change('start', pan_zoom_callback)
        p.x_range.js_on_change('end', pan_zoom_callback)

        # 9. Combine with the Reset button callback
        JS_CODE_RESET_LINES = """
            for (const k in lines) {
                lines[k].visible = true;
                circles[k].visible = true;
                if (areas[k]) {
                    areas[k].visible = true;
                }
            }
        """
        combined_reset_code = JS_CODE_RESET_LINES + "\n" + JS_CODE_DECIMATE_IMAGES
        combined_reset_args = dict(lines=lines, circles=circles, areas=areas, 
                                   all_images_source=all_images_source, 
                                   img_source=img_source, x_range=p.x_range)
        
        p.js_on_event(Reset, CustomJS(args=combined_reset_args, code=combined_reset_code))

        layout = column(p, img_plot, sizing_mode="stretch_both")
    else:
        # Fallback if no images were found
        layout = column(p, sizing_mode="stretch_both")

    output_file("keypoints_plot.html")
    show(layout)
    # --- End Image Subplot ---


# Example usage
if __name__ == "__main__":
    # Make sure to replace this with your actual folder path
    folders = ["images/NEONATE No 85"] 
    print("Processing folders:", folders)
    process_folders(folders)