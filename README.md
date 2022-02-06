# CV-Image-Stitching-Panaroma

Image Stitching Problem

**Make sure your OpenCV version is 3.4.2.17, as some of the SIFT features are now patented and not available in newer library.**

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing.

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints.
Next, you should match the keypoints in both images using the feature distance via KNN (k=2);
cross-checking and ratio test might be helpful for feature matching.
After this, you need to implement RANSAC algorithm to estimate homography matrix.
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image.

Only APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()” are used.

![imagestitching](screenshot1.png)
