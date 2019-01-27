import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import copy
from moviepy.editor import VideoFileClip

objpoints = []
imgpoints = []
nx = 9
ny = 6
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


def undistort(img):

    global objpoints
    global imgpoints
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst


def warp(undist, src, dst, w, h):
                 
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, (w,h))

    return warped, M, Minv


def combinedBinary(undistorted):

    thresh=(90, 255)
    s_thresh=(170, 255)
    sx_thresh=(20, 100)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary
    
def calibrateCamera():

    global objpoints
    global imgpoints

    
    objp = np.zeros((nx*ny, 3), np.float32 )
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    output_path = 'output_camera_cal/'
    path = 'camera_cal/'
    files = os.listdir(path)

    for name in files:
    
        # Make a list of calibration images
        fname = path+name
        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

            undistorted = undistort(img)

            offset = 100
            img_size = (gray.shape[1], gray.shape[0])
            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                            [img_size[0]-offset, img_size[1]-offset], 
                                            [offset, img_size[1]-offset]])

            # This is the warped chess board as a output for testing
            warped, _, _ = warp(undistorted, src, dst, img_size[0], img_size[1])

            plt.imsave(output_path+name, warped)

def findLanePixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fitPolynomial(binary_warped, M, IM, image, undist):
    
    global xm_per_pix
    global ym_per_pix

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = findLanePixels(binary_warped) 

    y_eval = 0

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0]) ## Implement the calculation of the left line here
    right_curverad =  ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])  ## Implement the calculation of the right line here
    
    center_offset = offsetFromCenter(leftx, lefty, rightx, righty, image.shape[1] )

    return left_fit, right_fit,left_curverad, right_curverad, center_offset

    

def offsetFromCenter(leftx,lefty, rightx, righty, frame_width):
    
    global xm_per_pix
    global ym_per_pix

    # found an awesome tutorial on computing this offset!
    left_bottom = np.mean(leftx[lefty > 0.95 * np.max(lefty)])
    right_bottom = np.mean(rightx[righty > 0.95 * np.max(righty)])

    lane_width = right_bottom - left_bottom
    midpoint = frame_width / 2

    offset_pix = abs((left_bottom + lane_width / 2) - midpoint)
    offset_meter = xm_per_pix * offset_pix

    return offset_meter

def fitLinesOnRoad(img_undistorted, Minv, left_fit, right_fit):
    
    height, width, _ = img_undistorted.shape

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # draw road as green polygon on original frame
    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))  # Warp back to original image space

    blend_onto_road = cv2.addWeighted(img_undistorted, 1., road_dewarped, 0.3, 0)

    def draw(img, fitting, color=(255, 0, 0), line_width=30):
       
        h, w, c = img.shape

        plot_y = np.linspace(0, h - 1, h)

        line_center = fitting[0] * plot_y ** 2 + fitting[1] * plot_y + fitting[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        # Some magic here to recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])


        # Draw the lane onto the warped blank image
        return cv2.fillPoly(img, [np.int32(pts)], color)

    # now separately draw solid lines to highlight them
    line_warp = np.zeros_like(img_undistorted)
    line_warp = draw(line_warp, left_fit,color=(255, 0, 0))
    line_warp = draw(line_warp, right_fit, color=(0, 0, 255))
    
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))


    lines_mask = blend_onto_road.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]

    blend_onto_road = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.)

    return blend_onto_road
   

def processPipeline(img): 

    # varibles to smoothen things out 
    left_fit_list = []
    right_fit_list = []

    left_radius_list = []
    right_radius_list = []

    lane_curvature = []

    # starting pipeline for processing images
    undistorted = undistort(img)

    combined_binary = combinedBinary(undistorted) 

    h, w = img.shape[:2]

    src = np.float32([[w, h-10],    
                    [0, h-10],    
                    [546, 460],   
                    [732, 460]])  
    dst = np.float32([[w, h],       
                    [0, h],       
                    [0, 0],       
                    [w, 0]]) 

    warped, M, Minv = warp(combined_binary, src, dst, w, h)  

    left_fit, right_fit, left_radius, right_radius, center_offset = fitPolynomial(warped, M, Minv, img, undistorted)

    left_fit_list.append(left_fit)
    right_fit_list.append(right_fit)

    left_radius_list.append(left_radius)
    right_radius_list.append(right_radius)

    final_output = fitLinesOnRoad(undistorted, Minv, np.mean(left_fit_list, axis=0), np.mean(right_fit_list, axis=0))

    mean_curvature = np.mean([np.mean(left_radius_list, axis=0), np.mean(right_radius_list, axis=0)])

    lane_curvature.append(mean_curvature)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_output, 'Lane curvature radius: {:.02f}m'.format(np.mean(lane_curvature)), (50, 60), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(final_output, 'Vehicle offset from center: {:.02f}m'.format(center_offset), (50, 130), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    return final_output

def testImagesProcessing():

    path = 'test_images/'
    files = os.listdir(path)

    output_path = "output_images/"

    for name in files:        

        img = mpimg.imread(path+name)

        final_output = processPipeline(img)

        plt.imsave(output_path+name, final_output)

def testVideosProcessing():

    videos = ["project_video.mp4","challenge_video.mp4","harder_challenge_video.mp4"]

    for video in videos:
        white_output = "output_videos/"+video

        print("Processing video " + video + " now.")
        clip1 = VideoFileClip(video)
        white_clip = clip1.fl_image(processPipeline) 
        white_clip.write_videofile(white_output, audio=False)
        
if __name__ == "__main__":

    """

    To run please run the following command:
        python3 main.py

    """

    print("Now calibrating our camera using a checkers board...\n")
    # calibrating the camera
    calibrateCamera()

    print("Now processing our test images and saving them to 'output_images' folder...\n")
    # output images saved in 'output_images' folder
    testImagesProcessing()

    print("Now processing our test videos and saving them to 'output_videos' folder...\n")
    # output images saved in 'output_videos' folder
    testVideosProcessing()