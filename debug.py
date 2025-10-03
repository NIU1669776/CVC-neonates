import os
import hashlib
import cv2
import numpy as np
from PIL import Image, ExifTags
from datetime import datetime

# Import your function (assumes it's on PYTHONPATH)
from temp_key_extraction import get_keypoint_temperature

def md5_of_file(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def pil_exif_orientation(path):
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        for k, v in ExifTags.TAGS.items():
            if v == 'Orientation':
                orient_tag = k
                break
        return exif.get(orient_tag, None)
    except Exception:
        return None

def save_side_by_side(a, b, outpath, label_a=None, label_b=None, scale=0.5):
    """
    Save a side-by-side image for quick visual inspection.
    a,b: BGR images as read by cv2 (or None)
    """
    if a is None and b is None:
        return
    # Convert to RGB for PIL use
    def cv2_to_pil(img):
        if img is None:
            return None
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        return pil
    pa = cv2_to_pil(a)
    pb = cv2_to_pil(b)
    # resize for easier viewing
    if pa is not None:
        pa = pa.resize((int(pa.width*scale), int(pa.height*scale)))
    if pb is not None:
        pb = pb.resize((int(pb.width*scale), int(pb.height*scale)))
    # create canvas
    w = (pa.width if pa else 0) + (pb.width if pb else 0)
    h = max((pa.height if pa else 0), (pb.height if pb else 0))
    canvas = Image.new('RGB', (w + 10, h + 40), (255,255,255))
    x = 0
    if pa:
        canvas.paste(pa, (x, 20))
        x += pa.width + 10
    if pb:
        canvas.paste(pb, (x, 20))
    # add labels
    import PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    if label_a:
        draw.text((5, 0), label_a, fill=(0,0,0), font=font)
    if label_b:
        draw.text((5 + (pa.width+10 if pa else 0), 0), label_b, fill=(0,0,0), font=font)
    canvas.save(outpath)

def compare_and_debug_folders(folder_a, folder_b, out_dir="debug_out", max_items=None):
    """
    Compare common thermal images between folder_a and folder_b, run the model on each pair,
    and save diagnostics into out_dir. Prints summary to stdout.
    """
    os.makedirs(out_dir, exist_ok=True)

    # gather thermal files (exclude .VIS.jpeg)
    def list_thermal(folder):
        files = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".jpeg") and not f.endswith(".VIS.jpeg"):
                files.append(f)
        return files

    a_files = list_thermal(folder_a)
    b_files = list_thermal(folder_b)

    common = sorted(list(set(a_files).intersection(set(b_files))))
    if not common:
        print("No common thermal filenames found between folders.")
        return

    if max_items:
        common = common[:max_items]

    summary = []
    for fname in common:
        print("\n=== Checking", fname, "===\n")
        ta = os.path.join(folder_a, fname)
        oa = ta.replace(".jpeg", ".VIS.jpeg")
        tb = os.path.join(folder_b, fname)
        ob = tb.replace(".jpeg", ".VIS.jpeg")

        # existence
        exists = {
            "ta": os.path.exists(ta),
            "oa": os.path.exists(oa),
            "tb": os.path.exists(tb),
            "ob": os.path.exists(ob)
        }
        print("Exists:", exists)
        if not (exists["ta"] and exists["oa"] and exists["tb"] and exists["ob"]):
            print("Missing one of the pair files; skipping detailed checks for this file.")
            summary.append((fname, "missing"))
            continue

        # md5 + filesize
        md5_ta = md5_of_file(ta)
        md5_oa = md5_of_file(oa)
        md5_tb = md5_of_file(tb)
        md5_ob = md5_of_file(ob)
        size_ta = os.path.getsize(ta)
        size_oa = os.path.getsize(oa)
        size_tb = os.path.getsize(tb)
        size_ob = os.path.getsize(ob)

        print(f"MD5 ta/tb: {md5_ta} / {md5_tb}")
        print(f"MD5 oa/ob: {md5_oa} / {md5_ob}")
        print(f"Sizes ta/tb: {size_ta} / {size_tb}")
        print(f"Sizes oa/ob: {size_oa} / {size_ob}")

        # read with OpenCV (UNCHANGED)
        img_ta_uc = cv2.imread(ta, cv2.IMREAD_UNCHANGED)
        img_oa_uc = cv2.imread(oa, cv2.IMREAD_UNCHANGED)
        img_tb_uc = cv2.imread(tb, cv2.IMREAD_UNCHANGED)
        img_ob_uc = cv2.imread(ob, cv2.IMREAD_UNCHANGED)

        # fallback read color
        img_ta = img_ta_uc if img_ta_uc is not None else cv2.imread(ta, cv2.IMREAD_COLOR)
        img_oa = img_oa_uc if img_oa_uc is not None else cv2.imread(oa, cv2.IMREAD_COLOR)
        img_tb = img_tb_uc if img_tb_uc is not None else cv2.imread(tb, cv2.IMREAD_COLOR)
        img_ob = img_ob_uc if img_ob_uc is not None else cv2.imread(ob, cv2.IMREAD_COLOR)

        def stats(img):
            if img is None:
                return None
            return {
                "shape": img.shape,
                "dtype": str(img.dtype),
                "min": float(np.min(img)),
                "max": float(np.max(img)),
                "mean": float(np.mean(img))
            }

        print("Stats ta:", stats(img_ta))
        print("Stats oa:", stats(img_oa))
        print("Stats tb:", stats(img_tb))
        print("Stats ob:", stats(img_ob))

        # EXIF orientation via PIL
        exif_ta = pil_exif_orientation(ta)
        exif_oa = pil_exif_orientation(oa)
        exif_tb = pil_exif_orientation(tb)
        exif_ob = pil_exif_orientation(ob)
        print("EXIF orientation (ta, oa, tb, ob):", exif_ta, exif_oa, exif_tb, exif_ob)

        # quick equality tests
        eq_thermal = False
        eq_vis = False
        try:
            eq_thermal = np.array_equal(img_ta, img_tb)
            eq_vis = np.array_equal(img_oa, img_ob)
        except Exception:
            pass
        print("Array-equal thermal:", eq_thermal, "vis:", eq_vis)

        # Save side-by-side visualizations
        s_out = os.path.join(out_dir, f"{fname}_thermal_compare.jpg")
        save_side_by_side(img_ta, img_tb, s_out, label_a=f"A:{folder_a}", label_b=f"B:{folder_b}")
        s_out2 = os.path.join(out_dir, f"{fname}_vis_compare.jpg")
        save_side_by_side(img_oa, img_ob, s_out2, label_a=f"A:{folder_a}", label_b=f"B:{folder_b}")

        # Run the model on both pairs
        try:
            res_a, dbg_a = get_keypoint_temperature(img_ta, img_oa)
            print("Result A keys:", list(res_a.keys()) if isinstance(res_a, dict) else res_a)
        except Exception as e:
            res_a, dbg_a = None, None
            print("Model error for A:", e)

        try:
            res_b, dbg_b = get_keypoint_temperature(img_tb, img_ob)
            print("Result B keys:", list(res_b.keys()) if isinstance(res_b, dict) else res_b)
        except Exception as e:
            res_b, dbg_b = None, None
            print("Model error for B:", e)

        # Save debug images returned by your function if any
        if dbg_a is not None:
            try:
                # if dbg_a is a numpy image
                dbg_path = os.path.join(out_dir, f"{fname}_A_dbg.jpg")
                if isinstance(dbg_a, np.ndarray):
                    cv2.imwrite(dbg_path, dbg_a)
                else:
                    # try PIL save if dbg is PIL
                    dbg_a.save(dbg_path)
                print("Saved A debug image to", dbg_path)
            except Exception as e:
                print("Could not save A debug image:", e)

        if dbg_b is not None:
            try:
                dbg_path = os.path.join(out_dir, f"{fname}_B_dbg.jpg")
                if isinstance(dbg_b, np.ndarray):
                    cv2.imwrite(dbg_path, dbg_b)
                else:
                    dbg_b.save(dbg_path)
                print("Saved B debug image to", dbg_path)
            except Exception as e:
                print("Could not save B debug image:", e)

        # If A produced a non-empty result and B did not, try re-saving the VIS from B with Pillow as baseline JPEG then re-run
        a_ok = isinstance(res_a, dict) and bool(res_a)
        b_ok = isinstance(res_b, dict) and bool(res_b)
        if a_ok and not b_ok:
            print("A succeeded but B failed → attempting to re-save B's VIS (to fix progressive/format issues) and re-run.")
            try:
                pil = Image.open(ob).convert("RGB")
                resave_path = os.path.join(out_dir, f"resaved_{os.path.basename(ob)}")
                pil.save(resave_path, format="JPEG", quality=95, progressive=False)
                print("Re-saved B VIS to", resave_path)
                # re-read and rerun
                img_ob_resaved = cv2.imread(resave_path, cv2.IMREAD_COLOR)
                res_b2, dbg_b2 = get_keypoint_temperature(img_tb, img_ob_resaved)
                print("After re-save, Result B keys:", list(res_b2.keys()) if isinstance(res_b2, dict) else res_b2)
                if dbg_b2 is not None:
                    dbg_path = os.path.join(out_dir, f"{fname}_B_dbg_after_resave.jpg")
                    if isinstance(dbg_b2, np.ndarray):
                        cv2.imwrite(dbg_path, dbg_b2)
                    else:
                        dbg_b2.save(dbg_path)
                    print("Saved B debug (after resave) to", dbg_path)
                # reflect success
                b_ok = isinstance(res_b2, dict) and bool(res_b2)
                res_b = res_b2
            except Exception as e:
                print("Resave attempt failed:", e)

        # record summary entry
        summary.append((fname, {
            "md5_ta": md5_ta, "md5_tb": md5_tb,
            "md5_oa": md5_oa, "md5_ob": md5_ob,
            "size_ta": size_ta, "size_tb": size_tb,
            "size_oa": size_oa, "size_ob": size_ob,
            "eq_thermal": eq_thermal, "eq_vis": eq_vis,
            "res_a_keys": list(res_a.keys()) if isinstance(res_a, dict) else None,
            "res_b_keys": list(res_b.keys()) if isinstance(res_b, dict) else None,
            "a_ok": a_ok, "b_ok": b_ok
        }))

    # Print compact summary
    print("\n=== SUMMARY ===")
    for item in summary:
        fname, info = item
        if isinstance(info, str):
            print(fname, "->", info)
        else:
            print(fname, "-> a_ok:", info["a_ok"], " b_ok:", info["b_ok"],
                  " eq_thermal:", info["eq_thermal"], " eq_vis:", info["eq_vis"])
    print("\nDiagnostics saved to:", os.path.abspath(out_dir))


# Example usage:
compare_and_debug_folders("Trial_folder", "Trial_folder_2", out_dir="debug_out", max_items=50)
