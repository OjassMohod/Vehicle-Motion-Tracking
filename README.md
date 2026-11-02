Vehicle Motion Tracking

What the program does is-
1) Creates a rectangle around the biggest moving object in the video using MOG2 background subtraction
2) Matches/Pairs points inside the rectangle between two consecutive frames of the video using ORB & SIFT feature matching along with Lowe's ration test & RANSAC to show more decisive matches only.
3)