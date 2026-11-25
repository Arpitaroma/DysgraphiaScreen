import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def parse_iamondb_xml(xml_file_path):
    """
    Reads an IamOnDB XML file and extracts continuous pen strokes.
    Returns (strokes, flat_trajectory) if points exist, otherwise (None, None).
    """
    if not os.path.exists(xml_file_path):
        return None, None

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError:
        return None, None

    strokes = []
    flat_rows = []

    # Search for Stroke or stroke tags (case-insensitive handling)
    stroke_elements = root.findall('.//Stroke') or root.findall('.//stroke')

    for stroke_idx, stroke_elem in enumerate(stroke_elements):
        points = []
        point_elements = stroke_elem.findall('Point') or stroke_elem.findall('point')
        
        for pt in point_elements:
            try:
                x = float(pt.attrib.get('x', 0))
                y = float(pt.attrib.get('y', 0))
                t = float(pt.attrib.get('time', 0))
                points.append([x, y, t])
                flat_rows.append([x, y, t, stroke_idx])
            except (ValueError, KeyError):
                continue
                
        if points:
            strokes.append(np.array(points, dtype=np.float32))

    if not strokes:
        return None, None

    flat_trajectory = np.array(flat_rows, dtype=np.float32)
    return strokes, flat_trajectory


def plot_strokes(strokes, title="Raw Handwriting Trajectory"):
    plt.figure(figsize=(9, 3))
    for stroke in strokes:
        plt.plot(stroke[:, 0], -stroke[:, 1], marker='.', markersize=2, linewidth=1.2)

    plt.title(title, fontsize=12)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    search_paths = [
        os.path.join("..", "data", "**", "*.xml"),
        os.path.join("data", "**", "*.xml")
    ]
    
    xml_files = []
    for pattern in search_paths:
        xml_files.extend(glob.glob(pattern, recursive=True))

    valid_file = None
    extracted_strokes = None
    extracted_flat = None

    # Iterate through XML files until one with actual stroke coordinates is found
    for file_path in xml_files:
        strokes, flat_data = parse_iamondb_xml(file_path)
        if strokes is not None and len(strokes) > 0:
            valid_file = file_path
            extracted_strokes = strokes
            extracted_flat = flat_data
            break

    if valid_file:
        print(f"Loaded valid trajectory file: {valid_file}")
        print(f"Strokes: {len(extracted_strokes)}, Total Points: {extracted_flat.shape[0]}")
        print(f"Sample data [x, y, time, stroke_id]:\n{extracted_flat[:5]}")
        plot_strokes(extracted_strokes, title=f"File: {os.path.basename(valid_file)}")
    else:
        print("No stroke trajectory XML files found. Please verify the unpacked data contents.")