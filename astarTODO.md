1. take in cropped satellite image
2. take in cropped path segment
3. take in image (from video/drone)
4. derive transformation matrix via opencv (other branch)
5. apply crowd weights to current matrix
6. calculate current frame's crowdness score & place onto path
7. scale by some factor (30% this frame, 70% historical data)
8. update historical data