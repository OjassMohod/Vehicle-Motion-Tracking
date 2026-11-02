# Vehicle Motion Tracking

## Plain-language summary

This program watches a traffic video and picks a moving object to follow. It draws a box around that object, follows visible details on it from one frame to the next, and shows the path it has taken. The video window also displays how far the box's center has moved from its starting position, measured in pixels. The program saves an animated GIF of the tracking and pictures showing how it compared details between two frames.

## Technical summary

The script reads `video3.mp4` with OpenCV and writes results to `outputs_simple/`.

1. **Moving-object detection:** MOG2 background subtraction processes the video. After a 30-frame warm-up, the foreground mask is thresholded above 200 to exclude shadows, then cleaned with morphological opening (3×3 kernel) and closing (5×5 kernel). External contours are filtered by area, bounding-box size, and aspect ratio. The largest qualifying bounding box in the first frame with candidates becomes the region of interest (ROI). The mask is saved as `foreground_mask.jpg`.
2. **Feature-matching demonstration:** The ROI in that frame is compared with a larger search area in the next frame using ORB and SIFT. Both crops are enlarged 3× before feature extraction. A brute-force matcher uses Hamming distance for ORB and L2 distance for SIFT. Lowe's 0.75 ratio test removes ambiguous matches; RANSAC identifies geometrically consistent matches when enough are available. The resulting visualizations are saved as `orb_matches.jpg` and `sift_matches.jpg`. These matches are for demonstration and do **not** drive the tracker.
3. **Tracking:** Shi–Tomasi corners are selected inside the ROI. Pyramidal Lucas–Kanade optical flow follows them through subsequent frames. Points with failed status, non-finite or out-of-frame positions, or motion more than 3 pixels from the median motion are discarded. The median displacement of the remaining points translates the fixed-size bounding box. Tracking stops if fewer than three usable points remain.
4. **Display and output:** Each tracked frame shows the bounding box, point-motion lines, center trajectory, and straight-line displacement of the box center from its initial position in pixels. The program displays the frames in an OpenCV window (press **Q** to stop) and saves sampled frames as `tracker_demo.gif`.
