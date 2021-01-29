import math
import colorsys

import cv2
import numpy as np


def get_color(tag, hue_step=0.41):
    tag = int(tag)
    h, v = (tag*hue_step) % 1, 1. - (int(tag*hue_step)%4)/5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(r*255), int(255*g), int(255*b)

def draw_bodypose25(frame, keypoints, thickness=3):
    """Draw single body pose on image frame

    Arguments:
        frame (np.ndarray): a RGB image frame
        keypoints (np.ndarray): array of size (25, 3) -> (x, y, score)
        thickness (int): limb thickness
    """
    limbs = [
            [1, 2], [1, 5], [1, 8], # Body
            [2, 3], [3, 4], # Left arm
            [5, 6], [6, 7], # Right arm
            [8, 9], [9, 10], [10, 11], # Left leg
            [11, 24], [11, 22], [22, 23], # Left ankle
            [8, 12], [12, 13], [13, 14], # Right leg
            [14, 21], [14, 19], [19, 20], # Right ankle
            [1, 0], # Neck
            [0, 15], [15, 17], # Lefy eye
            [0, 16], [16, 18] # Right eye
            ]
    limbs = (np.array(limbs)+1).tolist()
    colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 255, 0],
            [255, 255, 0], [255, 0, 170], [255, 0, 85], [255, 0, 10], [125, 0, 255],
            [125, 50, 85], [125, 50, 10], [125, 0, 170], [125, 0, 85], [255, 170, 0]
            ]

    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        if keypoint[-1] == 0:
            continue
        x = int(keypoint[0])
        y = int(keypoint[1])
        cv2.circle(frame, (x, y), 4, colors[i], thickness=-1)

    # Draw limbs
    cur_frame = frame.copy()
    for i, (limb, color) in enumerate(zip(limbs, colors)):
        keypointA = np.array(keypoints[limb[0]-1])
        keypointB = np.array(keypoints[limb[1]-1])
        if keypointA[-1] == 0 or keypointB[-1] == 0:
            continue

        keypointA = keypointA[:2]
        keypointB = keypointB[:2]
        mean = (keypointA + keypointB) / 2
        mean = mean.astype(np.int)
        length = np.sqrt(np.sum((keypointA-keypointB)**2))

        orientation = keypointB - keypointA
        angle = math.degrees(math.atan2(orientation[1], orientation[0]))
        polygon = cv2.ellipse2Poly(tuple(mean), (int(length / 2), thickness), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_frame, polygon, color)

    frame[:, :, :] = frame*0.4 + cur_frame*0.6
    return frame

def draw_bbox(frame, bbox, color=(85,135,255), thickness=3):
    """Draw bounding box on the specified frame

    Args:
        frame (ndarray): processing frame
        bbox (list): tlbr sequence of type list
        color (tuple): BGR color palette
        thickness (int): line thickness
    """
    tl_x, tl_y = tuple([ int(v) for v in bbox[:2]])
    br_x, br_y = tuple([ int(v) for v in bbox[2:4]])
    cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), color, thickness)

def draw_text(frame, text, position,
            fgcolor=(85, 125, 255),
            bgcolor=(85, 135, 255),
            fontScale=1, thickness=3, margin=5):
    """Draw text on the specified frame

    Args:
        frame (ndarray): processing frame
        text (string): text to render
        position (tuple): text position (tl_x, tl_y)
        fgcolor (tuple): BGR color palette for font color
        bgcolor (tuple): BGR color palette for background color
        fontScale (int): font scale
        thickness (int): line thickness
        margin (int): space between texts
    """
    # opencv doesn't handle `\n` in the text
    # therefore we handle it line by line
    lines = text.split('\n')
    text_widths = [ margin*2+cv2.getTextSize(text=line,
                                    thickness=thickness,
                                    fontScale=fontScale,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)[0][0]
                    for line in lines ]
    text_heights = [ margin*2+cv2.getTextSize(text=line,
                                    thickness=thickness,
                                    fontScale=fontScale,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)[0][1]
                    for line in lines ]
    max_width = int(max(text_widths))
    max_height = int(max(text_heights))
    tl_x = int(position[0])
    tl_y = int(position[1])

    # draw background
    cv2.rectangle(frame,
            (tl_x, tl_y),
            (tl_x+max_width, tl_y+max_height*len(lines)),
            bgcolor, -1)

    # draw text line by line
    for j, line in enumerate(lines):
        cv2.putText(frame, line,
                (tl_x+margin, tl_y+(max_height*(j+1))-margin),
                color=fgcolor,
                fontScale=fontScale,
                thickness=thickness,
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

def draw_gaussian(frame, mean, covariance, color=(85, 125, 255), thickness=3):
    vals, vecs = np.linalg.eigh(5.9915 * covariance)
    indices = vals.argsort()[::-1]
    vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

    center = int(mean[0] + .5), int(mean[1] + .5)
    axes = int(vals[0] + .5), int(vals[1] + .5)
    angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
    cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness)
