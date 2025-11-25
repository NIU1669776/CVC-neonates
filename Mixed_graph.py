import os, cv2
import numpy as np
from datetime import datetime
from temp_key_extraction import get_keypoint_temperature
from thermal import limit_finder
from bokeh.plotting import figure, show, output_file

# --- MODIFICATION: Clean, modern imports for Bokeh 3.8.1 ---
from bokeh.models import (
    HoverTool, ColumnDataSource, CustomJS, Range1d, Tabs, Select, TabPanel
)
# -----------------------------------------------------------

from bokeh.layouts import row, column
from bokeh.models.widgets import Button
from bokeh.palettes import Category10
import mediapipe as mp
from bokeh.events import Reset # Import Reset

mp_pose = mp.solutions.pose

# =============================================================================
# --- FUNCTIONS FOR PLOT 1: ABSOLUTE TEMPERATURE PLOT ---
# =============================================================================

def process_folders_temp(folders):
    """
    Process folders for the ABSOLUTE TEMPERATURE plot.
    Returns results with error: [(path, result_dict, error), ...]
    """
    pose = None
    results = []

    if len(folders) > 1:
        pose = mp_pose.Pose(static_image_mode=False)

    for folder in folders:
        for root, _, files in os.walk(folder):
            print(f"Processing folder (Temp): {root}")
            if len(files)>1 and pose is None:
                pose = mp_pose.Pose(static_image_mode=False)
            
            thermals = sorted([f for f in files if not f.endswith(".VIS.jpeg") and f.endswith(".jpeg")])
            originals = sorted([f for f in files if f.endswith(".VIS.jpeg")])

            for thermal in thermals:
                original_name = thermal.replace(".jpeg", ".VIS.jpeg")
                if original_name in originals:
                    thermal_path = os.path.join(root, thermal)
                    original_path = os.path.join(root, original_name)

                    thermal_img = cv2.imread(thermal_path)
                    original_img = cv2.imread(original_path)
                    
                    TOP,BOT = limit_finder(thermal_img)
                    error = min((TOP-BOT)*0.02,2)
                    result, _ = get_keypoint_temperature(thermal_img, original_img, pose)
                    
                    results.append((thermal_path, result, error)) 

    results.sort(key=lambda x: x[0])
    print(f"Total processed for Temp plot: {len(results)}")

    if not results:
        print("No results for Temp plot.")
        return None
    return results


def plot_temp(results):
    """
    Plot the ABSOLUTE TEMPERATURE results with dynamic images.
    Returns a 'layout' object.
    """

    # --- Data Prep ---
    keypoints = []
    for _, res, _ in results: 
        if isinstance(res, dict) and res:
            keypoints = list(res.keys())
            break
    if not keypoints:
        print("No keypoints found for Temp plot.")
        return None

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
    # --- End Data Prep ---

    # --- Main plot ---
    p = figure(title="Temperature Keypoints Over Time (Interactive)",
               x_axis_type="datetime", x_axis_label="Timestamp", y_axis_label="Temperature",
               sizing_mode="stretch_both", tools="pan,wheel_zoom,box_zoom,reset,save")

    palette = Category10[10] if len(keypoints) <= 10 else (Category10[10] * ((len(keypoints)//10)+1))
    lines, circles, areas = {}, {}, {}

    for i, keypoint in enumerate(keypoints):
        color = palette[i % len(palette)]
        legend_col_name = f"{keypoint}_name"

        areas[keypoint] = p.varea(
            x='timestamps', y1=f"{keypoint}_lower", y2=f"{keypoint}_upper", source=source,
            fill_color=color, fill_alpha=0.5, name=keypoint, visible=True, legend_field=legend_col_name
        )
        lines[keypoint] = p.line(
            'timestamps', keypoint, source=source, line_width=2, color=color,
            name=keypoint, visible=True, legend_field=legend_col_name
        )
        circles[keypoint] = p.circle(
            'timestamps', keypoint, source=source, size=6, color=color, alpha=0.9,
            line_color=None, name=keypoint, visible=True, legend_field=legend_col_name
        )

    hover = HoverTool(
        tooltips=[
            ("Keypoint", "$name"), ("Temperature", f"$y (Error: +/- @error{{0.00}})"),
            ("Timestamp", "@timestamps{%F %T}"),
        ],
        formatters={"@timestamps": "datetime"}, mode="mouse", renderers=list(circles.values())
    )
    p.add_tools(hover)

    p.legend.click_policy = "hide"
    p.legend.title = "Click to toggle keypoints"
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "10pt"
    # --- End Main plot ---

    # --- Image Subplot (Dynamic) ---
    all_urls, all_xs, all_ys = [], [], []
    for i, (path, _, _) in enumerate(results): 
        original_path = path.replace(".jpeg", ".VIS.jpeg")
        dt = timestamps[i]
        if os.path.exists(original_path) and dt is not None:
            all_urls.append(original_path)
            all_xs.append(dt)
            all_ys.append(0.5)

    if all_urls:
        all_images_source = ColumnDataSource(data=dict(url=all_urls, x=all_xs, y=all_ys))
        
        initial_x, initial_url, initial_w, initial_y = [], [], [], []
        target_images = 10
        
        timestamps_ms = [dt.timestamp() * 1000 for dt in all_xs]
        if timestamps_ms:
            start_ms, end_ms = timestamps_ms[0], timestamps_ms[-1]
            view_duration_ms = end_ms - start_ms
            
            new_width_ms = (view_duration_ms / target_images) * 0.9
            
            step = max(1, len(all_xs) // target_images)
            for i in range(0, len(all_xs), step):
                initial_x.append(all_xs[i])
                initial_url.append(all_urls[i])
                initial_w.append(new_width_ms)
                initial_y.append(0.5)
        
        img_source = ColumnDataSource(data=dict(url=initial_url, x=initial_x, y=initial_y, w=initial_w))

        img_plot = figure(x_axis_type="datetime", x_range=p.x_range, y_range=Range1d(0, 1),
                          height=180, sizing_mode="stretch_width", toolbar_location=None)
        
        img_plot.image_url(url="url", x="x", y="y", w='w', h=0.8, anchor="center", source=img_source)
        
        img_plot.yaxis.visible = False
        img_plot.xaxis.visible = False
        img_plot.grid.visible = False
        
        JS_CODE_DECIMATE_IMAGES = """
            const all_data = all_images_source.data;
            const img_data = img_source.data;
            const all_x = all_data['x'];
            const all_url = all_data['url'];
            
            const start = x_range.start, end = x_range.end;
            const view_duration = end - start;

            const TARGET_IMAGES = 10;
            const new_width = (view_duration / TARGET_IMAGES) * 0.9;

            const new_x = [], new_url = [], new_w = [], new_y = [];

            const indices = [];
            for (let i = 0; i < all_x.length; i++) {
                if (all_x[i] >= start - new_width/2 && all_x[i] <= end + new_width/2) {
                    indices.push(i);
                }
            }

            const step = Math.max(1, Math.floor(indices.length / TARGET_IMAGES));
            for (let i = 0; i < indices.length; i += step) {
                let idx = indices[i];
                new_x.push(all_x[idx]);
                new_url.push(all_url[idx]);
                new_w.push(new_width);
                new_y.push(0.5); 
            }
            img_source.data = {'x': new_x, 'url': new_url, 'w': new_w, 'y': new_y};
        """
        
        pan_zoom_callback = CustomJS(
            args=dict(all_images_source=all_images_source, img_source=img_source, x_range=p.x_range),
            code=JS_CODE_DECIMATE_IMAGES
        )
        p.x_range.js_on_change('start', pan_zoom_callback)
        p.x_range.js_on_change('end', pan_zoom_callback)

        JS_CODE_RESET_LINES = """
            for (const k in lines) {
                lines[k].visible = true;
                circles[k].visible = true;
                if (areas[k]) { areas[k].visible = true; }
            }
        """
        combined_reset_code = JS_CODE_RESET_LINES + "\n" + JS_CODE_DECIMATE_IMAGES
        combined_reset_args = dict(lines=lines, circles=circles, areas=areas, 
                                   all_images_source=all_images_source, 
                                   img_source=img_source, x_range=p.x_range)
        
        p.js_on_event(Reset, CustomJS(args=combined_reset_args, code=combined_reset_code))

        layout = column(p, img_plot, sizing_mode="stretch_both")
    else:
        layout = column(p, sizing_mode="stretch_both")

    return layout
    # --- End Image Subplot ---

# =============================================================================
# --- FUNCTIONS FOR PLOT 2: TEMPERATURE DIFFERENCE PLOT ---
# =============================================================================

def process_folders_diff(folders):
    """
    Process folders for the TEMPERATURE DIFFERENCE plot.
    Returns results with error: [(path, result_dict, error), ...]
    """
    pose = None
    results = []

    if len(folders) > 1:
        pose = mp_pose.Pose(static_image_mode=False)

    for folder in folders:
        for root, _, files in os.walk(folder):
            print(f"Processing folder (Diff): {root}")
            if len(files) > 1 and pose is None:
                pose = mp_pose.Pose(static_image_mode=False)

            thermals = sorted([f for f in files if not f.endswith(".VIS.jpeg") and f.endswith(".jpeg")])
            originals = sorted([f for f in files if f.endswith(".VIS.jpeg")])

            for thermal in thermals:
                original_name = thermal.replace(".jpeg", ".VIS.jpeg")
                if original_name in originals:
                    thermal_path = os.path.join(root, thermal)
                    original_path = os.path.join(root, original_name)

                    thermal_img = cv2.imread(thermal_path)
                    original_img = cv2.imread(original_path)
                    
                    TOP,BOT = limit_finder(thermal_img)
                    error = min((TOP-BOT)*0.02,2)
                    result, _ = get_keypoint_temperature(thermal_img, original_img, pose)
                    
                    results.append((thermal_path, result, error))

    results.sort(key=lambda x: x[0])
    print(f"Total processed for Diff plot: {len(results)}")
    
    if not results:
        print("No results for Diff plot.")
        return None
    return results


def plot_diff(results):
    """
    Plot the temperature difference between two selectable keypoints over time,
    with combined error bands. Returns a 'layout' object.
    """
    # --- Extract keypoints and timestamps ---
    keypoints = []
    for _, res, _ in results: # Unpack 3 items
        if isinstance(res, dict) and res:
            keypoints = list(res.keys())
            break

    if not keypoints:
        print("No keypoints found for Diff plot.")
        return None

    timestamps = []
    for path, _, _ in results: # Unpack 3 items
        timestamps.append(_extract_datetime_from_filename(path))

    # --- Build data columns with error ---
    data = {'timestamps': timestamps}
    errors = []
    for _, _, err_val in results:
        errors.append(_safe_val(err_val))
    data['error'] = errors # Add the single error column

    for keypoint in keypoints:
        col = []
        for _, res, _ in results:
            val = res.get(keypoint) if isinstance(res, dict) else None
            col.append(_safe_val(val))
        data[keypoint] = col
    # ---------------------------------------------------

    # Initialize the data sources
    source = ColumnDataSource(data=data)
    # --- Add columns for error bands ---
    diff_source_data = dict(
        timestamps=timestamps, 
        diff=[np.nan] * len(timestamps),
        upper=[np.nan] * len(timestamps),
        lower=[np.nan] * len(timestamps),
        combined_err=[np.nan] * len(timestamps)
    )
    diff_source = ColumnDataSource(data=diff_source_data)
    # -------------------------------------------------

    # --- Create the figure ---
    p = figure(title="Temperature Difference Between Two Keypoints",
               x_axis_type="datetime",
               x_axis_label="Timestamp",
               y_axis_label="Temperature Difference (°C)",
               sizing_mode="stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset,save")

    # --- Add varea glyph for error ---
    p.varea(x='timestamps', y1='lower', y2='upper', source=diff_source,
            fill_color="firebrick", fill_alpha=0.3, legend_label="Combined Error")
    # -----------------------------------------------
    
    diff_line = p.line('timestamps', 'diff', source=diff_source, line_width=3,
                       color="firebrick", legend_label="Difference")
    diff_circle = p.circle('timestamps', 'diff', source=diff_source, size=6,
                           color="firebrick", alpha=0.8, legend_label="Difference")
    
    p.legend.click_policy = "hide"

    # --- Update HoverTool tooltips ---
    hover = HoverTool(
    tooltips="""
        <div>
            <strong>@{s1.value}</strong> vs <strong>@{s2.value}</strong><br>
            <span style="color:firebrick;">
                Diff: <b>@{diff}{0.00} °C</b><br>
                Error: <b>± @{combined_err}{0.00} °C</b>
            </span><br>
            <em>@timestamps{%F %T}</em>
        </div>
    """,
    formatters={"@timestamps": "datetime"},
    mode="mouse",
    renderers=[diff_circle],
    )
    p.add_tools(hover)
    # ---------------------------------------------

    # --- Keypoint selectors ---
    select1 = Select(title="Keypoint 1 (T1)", value=keypoints[0], options=keypoints)
    select2 = Select(title="Keypoint 2 (T2)", value=keypoints[1] if len(keypoints) > 1 else keypoints[0], options=keypoints)

    # --- JS Callback to calculate error bands ---
    callback = CustomJS(args=dict(src=source, diff_src=diff_source, s1=select1, s2=select2), code="""
        const data = src.data;
        const t = data['timestamps'];
        const k1 = s1.value;
        const k2 = s2.value;
        const y1 = data[k1];
        const y2 = data[k2];
        const errors = data['error']; // Get the error column
        
        const diff_data = diff_src.data;
        const diff = [];
        const upper = [];
        const lower = [];
        const combined_err = [];

        for (let i = 0; i < t.length; i++) {
            const v1 = y1[i];
            const v2 = y2[i];
            const e = errors[i];
            
            if (isNaN(v1) || isNaN(v2) || isNaN(e)) {
                diff.push(NaN);
                upper.push(NaN);
                lower.push(NaN);
                combined_err.push(NaN);
            } else {
                const diff_val = v1 - v2;
                // Combined error is e1 + e2. Assuming e1 = e2 = e from the image.
                const comb_err = e + e; 
                
                diff.push(diff_val);
                upper.push(diff_val + comb_err);
                lower.push(diff_val - comb_err);
                combined_err.push(comb_err);
            }
        }
        diff_data['timestamps'] = t;
        diff_data['diff'] = diff;
        diff_data['upper'] = upper;
        diff_data['lower'] = lower;
        diff_data['combined_err'] = combined_err;
        
        diff_src.change.emit();
    """)
    # --------------------------------------------------------

    select1.js_on_change("value", callback)
    select2.js_on_change("value", callback)

    # Trigger initial computation
    callback.args["s1"].value = select1.value
    callback.args["s2"].value = select2.value
    callback.code += "\n" + "s1.change.emit();"

    layout = column(row(select1, select2, sizing_mode="stretch_width"), p, sizing_mode="stretch_both")

    return layout


# =============================================================================
# --- MAIN EXECUTION ---
# =============================================================================

def _safe_val(v):
    """(Helper) Return a float or np.nan for invalid/missing values."""
    if v is None: return np.nan
    try: f = float(v)
    except (TypeError, ValueError): return np.nan
    return f if np.isfinite(f) else np.nan

def _extract_datetime_from_filename(path):
    """(Helper) Extract datetime from filenames like HM20241025142419.jpeg"""
    fname = os.path.basename(path)
    try:
        idx = fname.index("HM")
        timestamp_str = fname[idx + 2:idx + 16]
        dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        return dt
    except Exception as e:
        print(f"Warning: could not parse datetime from {fname} ({e})")
        return None


if __name__ == "__main__":
    # Define the folders to process. Both plots will use this same data.
    folders = ["images/NEONATE No 85"]
    
    tabs_list = []
    
    # --- 1. Generate Temperature Plot ---
    print("--- Processing Plot 1 (Temperature) ---")
    results_1 = process_folders_temp(folders)
    if results_1:
        plot_1 = plot_temp(results_1)
        if plot_1:
            # --- MODIFICATION: Use TabPanel ---
            tab1 = TabPanel(child=plot_1, title="Absolute Temperature")
            # --------------------------------
            tabs_list.append(tab1)

    # --- 2. Generate Difference Plot ---
    print("\n--- Processing Plot 2 (Difference) ---")
    results_2 = process_folders_diff(folders) 
    if results_2:
        plot_2 = plot_diff(results_2)
        if plot_2:
            # --- MODIFICATION: Use TabPanel ---
            tab2 = TabPanel(child=plot_2, title="Temperature Difference")
            # --------------------------------
            tabs_list.append(tab2)

    # --- 3. Show Final Tabbed Layout ---
    if tabs_list:
        final_layout = Tabs(tabs=tabs_list, sizing_mode="stretch_both")
        output_file("keypoints_dashboard.html")
        print("\nSuccessfully generated 'keypoints_dashboard.html'")
        show(final_layout)
    else:
        print("\nNo plots were generated. No data found.")