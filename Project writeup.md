## Jeyte Hagaley's Project Writeup

### This my write up for the Advanced Lane Lines project, below are the details for each portion of this project.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/test3.jpg "Output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! This document serves as my write up and includes a review of all of the work I completed to accomplish this project!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 'main.py' file in the `calibrateCamera()` function. The calibration of the camera works as follows.   

I start by preparing finding the corners of the chessboard for each image. The store the "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

After calibrating the camera I was ready to start processing road images!

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Using the `objpoints` and `imgpoints` arrays from the camera calibration I was easily able to undistort images using the `cv2.undistort()` function. This was the first step of the pipeline and was greatly simplified thanks to the previous step of camera calibration. 

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in my `combinedBinary()` function. This function took in an undistored image and using the techniques we learned in class such as using HLS color spaces and Sobel transform I was able to get a get a combined binary image that had incorporated both of the above techniques and combined the threshold channels.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the file `main.py`. The `warp()` function takes as inputs an source (`src`) and destination (`dst`) points, the image width and height (`w,h`) and the undistored image(`undist`). I chose the source and destination points in the following manner in the this function as:

```python

h, w = img.shape[:2]

src = np.float32([[w, h-10],    
                [0, h-10],    
                [546, 460],   
                [732, 460]])  
dst = np.float32([[w, h],       
                [0, h],       
                [0, 0],       
                [w, 0]]) 
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The results can be seen from the output images saved from the camera calibration step in the `output_camera_cal` folder.



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After taking the perspective transform of the image I then used  the windowing approach to first create a histogram of the peaks of the combined image. Following the example in class using 9 for my `nwindows` I was able to get back points to mark for my left and right lan lines. Then I used these points to fit a 2nd order polynomial to approximate the curvature of the each lane. Below is a visual representation of what was calculated:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Both of these steps were completed in my `fitPolynomial()` function. I used a separate method called `offsetFromCenter()` to compute how far away the vehicle was from center but computed the radius of curvature for both the left and right lane lines in the `fitPolynomial()`function.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This was the final step of my project and was completed in the `fitLinesOnRoad()` function in `main.py`. Using an Inner method name `draw()` I was able to take the original image, a fitting for the lane line, and the color and I was able to draw the lane on the road. Below is the function I used to accomplish this:

```python

def draw(img, fitting, color=(255, 0, 0), line_width=30):
       
    h, w, c = img.shape

    plot_y = np.linspace(0, h - 1, h)

    line_center = fitting[0] * plot_y ** 2 + fitting[1] * plot_y + fitting[2]
    line_left_side = line_center - line_width // 2
    line_right_side = line_center + line_width // 2

    pts_left = np.array(list(zip(line_left_side, plot_y)))
    pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
    pts = np.vstack([pts_left, pts_right])


    # Drawing the lane onto the warped blank image
    return cv2.fillPoly(img, [np.int32(pts)], color)
```


## Below you can see an example of one of my output images!
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here how I might improve my project if I were going to pursue this project further. The first thing that I would like to improve is my calculation for the curvature of the road. This was a very difficult part of the project for me and I did the best I could. Another improvement is the speed of my pipeline, I am not sure how long a video should take to process on average but I would have loved for it to be a little faster. The final improvement that comes to mind is improving my pipeline to better respond to the challenge videos provided. I think my code had a very hard time responding to multiple curves but I believe I could solve this using a 3rd or even a 4th order polynomial when computing my lanes. 
