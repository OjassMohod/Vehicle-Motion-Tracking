from pathlib import Path
import cv2
import numpy as np
from PIL import Image

FOLDER = Path(__file__).resolve().parent
VIDEO = FOLDER / "video3.mp4"
OUTPUT = FOLDER / "outputs_simple"
WARMUP_FRAMES = 30
MIN_AREA = 300 

def detect_moving_candidate(video, first):
    background = cv2.createBackgroundSubtractorMOG2(
        history=150, varThreshold=25, detectShadows=True
    )
    background.apply(first)
    frame_number = 1

    while True:
        success, frame = video.read()
        if not success:
            raise RuntimeError("No moving candidate found. Try a stationary-camera traffic clip.")

        foreground = background.apply(frame)
        frame_number += 1
        if frame_number <= WARMUP_FRAMES:
            continue

        _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        contours, _ = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        image_area = frame.shape[0] * frame.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if MIN_AREA <= area <= image_area * 0.2 and w >= 12 and h >= 12 and 0.3 <= w / h <= 4:
                candidates.append((x, y, w, h))

        if candidates:
            box = max(candidates, key=lambda b: b[2] * b[3])
            cv2.imwrite(str(OUTPUT / "foreground_mask.jpg"), foreground)
            print(f"Selected moving candidate at frame {frame_number}: {box}")
            return frame, box


def demonstrate_matching(first, second, box, method):
    # Comparing the selected object in two consecutive frames using ORB or SIFT
    x, y, w, h = box
    image_height, image_width = first.shape[:2]

    crop1 = first[y:y + h, x:x + w]
    crop2 = second[max(0, y - h):min(image_height, y + 2 * h),
                   max(0, x - w):min(image_width, x + 2 * w)]

    crop1 = cv2.resize(crop1, None, fx=3, fy=3)
    crop2 = cv2.resize(crop2, None, fx=3, fy=3)
    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=300, edgeThreshold=8,
                                  patchSize=15, fastThreshold=8)
        distance = cv2.NORM_HAMMING
    else:
        detector = cv2.SIFT_create(nfeatures=300)
        distance = cv2.NORM_L2

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    good_matches = []

    if descriptors1 is not None and descriptors2 is not None and len(descriptors2) >= 2:
        matcher = cv2.BFMatcher(distance)
        pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)

        for pair in pairs:
            if len(pair) == 2:
                best, next_best = pair
                if best.distance < 0.75 * next_best.distance:
                    good_matches.append(best)

    # RANSAC
    verified_matches = []
    if len(good_matches) >= 4:
        old = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        new = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        matrix, inliers = cv2.estimateAffinePartial2D(
            old, new, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if matrix is not None and inliers is not None and inliers.sum() >= 4:
            verified_matches = [m for m, keep in zip(good_matches, inliers.ravel()) if keep]

    picture = cv2.drawMatches(
        crop1, keypoints1, crop2, keypoints2, verified_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(OUTPUT / f"{method.lower()}_matches.jpg"), picture)
    message = f"{method}: {len(good_matches)} ratio-test matches, {len(verified_matches)} RANSAC inliers."
    print(message)
    if not verified_matches:
        print("  Too few reliable matches: try a video with a larger, sharper vehicle.")
    return message


def main():
    OUTPUT.mkdir(exist_ok=True)
    video = cv2.VideoCapture(str(VIDEO))
    gif_frames = []

    try:
        success, first = video.read()
        if not success:
            raise RuntimeError("Could not read video3.mp4. Check its location.")

        fps = video.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30
        delay = max(1, round(1000 / fps))
        gif_step = max(1, round(fps / 8))

        first, box = detect_moving_candidate(video, first)

        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        image_height, image_width = first.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_width, x2), min(image_height, y2)
        x, y = float(x1), float(y1)
        w, h = x2 - x1, y2 - y1

        #DEMONSTRATE ORB AND SIFT ON TWO FRAMES
        success, frame = video.read()
        if not success:
            raise RuntimeError("The video needs at least two frames.")
        box = (x1, y1, w, h)
        orb_message = demonstrate_matching(first, frame, box, "ORB")
        sift_message = demonstrate_matching(first, frame, box, "SIFT")

        #INITIALISE CORNERS FOR LUCAS-KANADE TRACKING
        previous_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(previous_gray)
        mask[y1:y2, x1:x2] = 255
        points = cv2.goodFeaturesToTrack(previous_gray, mask=mask,
                                        maxCorners=50, qualityLevel=0.01,
                                        minDistance=3)
        if points is None or len(points) < 3:
            raise RuntimeError("Too few corners on this vehicle. Try a clearer clip.")

        start_center = np.array([x + w / 2, y + h / 2])
        trail = [tuple(np.rint(start_center).astype(int))]
        frame_number = 0
        print("Tracking started. Press Q in the video window to stop.")

        #TRACKING THE VEHICLE THROUGH THE REMAINING VIDEO
        while success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                previous_gray, gray, points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            if new_points is None or status is None:
                print("Tracking lost.")
                break

            valid = status.ravel() == 1
            old = points.reshape(-1, 2)[valid]
            new = new_points.reshape(-1, 2)[valid]
            valid = (np.isfinite(new).all(axis=1)
                     & (new[:, 0] >= 0) & (new[:, 0] < image_width)
                     & (new[:, 1] >= 0) & (new[:, 1] < image_height))
            old, new = old[valid], new[valid]
            if len(new) < 3:
                print("Too few tracked points remain.")
                break

            movement = new - old
            typical_movement = np.median(movement, axis=0)
            consistent = np.linalg.norm(movement - typical_movement, axis=1) < 3
            old, new = old[consistent], new[consistent]
            if len(new) < 3:
                print("Too few consistent points remain.")
                break

            dx, dy = np.median(new - old, axis=0)
            x += dx
            y += dy
            vehicle_center = np.array([x + w / 2, y + h / 2])
            displacement = np.linalg.norm(vehicle_center - start_center)
            trail.append(tuple(np.rint(vehicle_center).astype(int)))

        #DRAWING THE BOX, POINTS, TRAJECTORY AND DISPLACEMENT
            output = frame.copy()
            cv2.rectangle(output, (round(x), round(y)),
                          (round(x + w), round(y + h)), (0, 255, 0), 2)
            for p, q in zip(old, new):
                p = tuple(np.rint(p).astype(int))
                q = tuple(np.rint(q).astype(int))
                cv2.line(output, p, q, (0, 255, 255), 1)
                cv2.circle(output, q, 2, (0, 255, 0), -1)
            cv2.polylines(output, [np.int32(trail)], False, (255, 0, 255), 2)
            cv2.putText(output, f"Displacement: {displacement:.1f} px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if frame_number % gif_step == 0:
                scale = min(1.0, 640 / output.shape[1])
                small = cv2.resize(output, None, fx=scale, fy=scale)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                gif_frames.append(Image.fromarray(rgb).quantize(colors=128))

            cv2.imshow("Vehicle Tracker", output)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

            previous_gray = gray
            points = new.reshape(-1, 1, 2).astype(np.float32)
            success, frame = video.read()
            frame_number += 1

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        video.release()
        cv2.destroyAllWindows()

        #SAVE THE GIF
        if gif_frames:
            gif_frames[0].save(
                OUTPUT / "tracker_demo.gif", save_all=True,
                append_images=gif_frames[1:],
                duration=max(10, round(1000 * gif_step / fps)), loop=0
            )
            print("Saved outputs_simple/tracker_demo.gif and matching pictures.")

if __name__ == "__main__":
    main()
