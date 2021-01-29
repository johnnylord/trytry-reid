import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname((osp.abspath(__file__)))))
import argparse

import cv2
import numpy as np
import pandas as pd

from utils.display import draw_bodypose25, draw_text, draw_bbox


WIN_SIZE = (760, 1080)

SEMANTIC_LABEL = {
    -1: "Unknown",
    0: "North",
    1: "North East",
    2: "East",
    3: "South East",
    4: "South",
    5: "South West",
    6: "West",
    7: "North West",
    }

TOTAL_LENGTH = -1
CURRENT_INDEX = 0

def trackbar_callback(value):
    global CURRENT_INDEX
    CURRENT_INDEX = value

def main(args):
    img_dir = args['dir']
    img_files = [ osp.join(img_dir, f) for f in os.listdir(img_dir) ]

    # Create display window
    cv2.namedWindow("Display", cv2.WINDOW_GUI_EXPANDED)
    cv2.createTrackbar("Index", "Display", 0, len(img_files)-1, trackbar_callback)
    if args['hint']:
        cv2.namedWindow("Hint", cv2.WINDOW_GUI_EXPANDED)
        path = f'{osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "imgs/hint.png")}'
        hint = cv2.imread(path)

    # Create labels placeholder
    labels = dict([ (osp.basename(f), -1) for f in img_files ])
    if osp.exists(args['history']):
        df = pd.read_csv(args['history'])
        for _, row in df.iterrows():
            name = row['name']
            label = row['label']
            labels[name] = label

    names = sorted(list(labels.keys()))
    global CURRENT_INDEX
    while CURRENT_INDEX < len(names):
        # Load target image and current label
        name = names[CURRENT_INDEX]
        path = osp.join(img_dir, name)
        img = cv2.imread(path)
        if img is None:
            break
        label = labels[name]
        # Resize image
        old_size = img.shape[:2][::-1]
        new_size = (WIN_SIZE[0]//2, WIN_SIZE[1])
        x_scale, y_scale = (np.array(new_size)/np.array(old_size)).tolist()
        img = cv2.resize(img, new_size)
        # Draw current label
        draw_text(img, f"Label:{label}, {SEMANTIC_LABEL[label]}",
                position=(0, 0),
                fgcolor=(255, 255, 255),
                bgcolor=(0, 0, 255))
        # Show image
        cv2.imshow("Display", img)
        cv2.setTrackbarPos("Index", "Display", CURRENT_INDEX)
        if osp.exists(args['hint']):
            cv2.imshow("Hint", hint)

        # Key handler
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif 48 <= key <= 55: # 0 - 7
            label = key - 48
            labels[name] = label
            print(f"Label: {SEMANTIC_LABEL[label]}")
            CURRENT_INDEX += 1
        elif key == 83: # Right arrow
            CURRENT_INDEX += 1 if CURRENT_INDEX < len(names) else 0
        elif key == 81: # Left arrow
            CURRENT_INDEX -= 1 if CURRENT_INDEX > 0 else 0
        elif key == 32: # Space
            next_index = CURRENT_INDEX
            while True:
                try:
                    next_name = names[next_index]
                    next_label = labels[next_name]
                    if next_label == -1:
                        CURRENT_INDEX = next_index
                        break
                    next_index += 1
                except:
                    break

    # Export label data (orient.csv)
    df = pd.DataFrame({ 'name': labels.keys(),
                        'label': labels.values() })
    df.to_csv(args['output'], index=False)
    print("Save label data to '{}'".format(args['output']))
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="directory containing images")
    parser.add_argument("--output", default="orient.csv", help="output file")
    parser.add_argument("--history", default="", help="prelabeled orient.csv")
    parser.add_argument("--hint", action='store_true', help="show label hint")

    args = vars(parser.parse_args())
    main(args)
