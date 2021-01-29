import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import argparse

import io
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import grpc
import message.pose_estimation_pb2 as pose_estimation_pb2
import service.pose_estimation_pb2_grpc as pose_estimation_pb2_grpc

from utils.display import draw_bodypose25


def main(args):
    # Establish connection to server and interact with it
    with grpc.insecure_channel(f"{args['ip']}:{args['port']}") as channel:
        # Construct grpc service client handler
        stub = pose_estimation_pb2_grpc.EstimationStub(channel)

        # Aggregate raw images
        files = [ osp.join(args['input'], f)
                for f in os.listdir(args['input'])
                if 'jpg' in f ]

        # Processing raw images
        for f in tqdm(files):
            # Read image source
            img = Image.open(f)
            buf = io.BytesIO()
            img.save(buf, format=img.format)

            # Construct service request
            request = pose_estimation_pb2.EstimateRequest()
            request.img.payload = buf.getvalue()

            # Use gRPC service
            response = stub.EstimatePoses(request)

            # Filter out pose images with multiple pose descriptions
            if len(response.poseKeypoints) != 1:
                continue

            # Extract keypoints
            keypoints = np.array([
                            (p.x, p.y, p.conf)
                            for p in response.poseKeypoints[0].points ])

            # Filter out pose images with fews valid keypoints
            if np.sum(keypoints[:, -1] > 0) < 10:
                continue

            # Draw keypoints
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            canvas = np.zeros_like(img, dtype=np.uint8)
            draw_bodypose25(img, keypoints=keypoints, thickness=2)
            draw_bodypose25(canvas, keypoints=keypoints, thickness=2)

            # Export to output directory
            if not osp.exists(args['output']):
                os.makedirs(args['output'])
            fname = osp.join(args['output'], osp.basename(f))
            cv2.imwrite(fname, canvas)

            # Display detected objects
            if args['gui']:
                cv2.namedWindow("Display", cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("Display", img)
                key = cv2.waitKey(500)
                if key == ord('q'):
                    break

        if args['gui']:
            cv2.destroyAllWindows()

        # Show processing result
        n_raws = len([ f for f in os.listdir(args['input']) if 'jpg' in f ])
        n_poses = len([ f for f in os.listdir(args['output']) ])
        print(f"{int(n_poses/n_raws*100)}% of valid poses are extracted from raw dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="localhost", help="server ip")
    parser.add_argument("--port", default=50000, help="server port")
    parser.add_argument("--input", required=True, help="img to be processed")
    parser.add_argument("--output", default="poses", help="output directory for pose imgs")
    parser.add_argument("--gui", action='store_true', help="Show gui result")

    args = vars(parser.parse_args())
    main(args)
