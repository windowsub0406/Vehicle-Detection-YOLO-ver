import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from find_car import *
import time
input_type = 'image' #'video' # 'image'
input_name = 'test_images/test1.jpg' #'test_images/test1.jpg' # 'project_video.mp4'

if __name__ == '__main__':
    if input_type == 'image':
        image = cv2.imread(input_name)
        draw_image = np.copy(image)
        y_start_stop1, y_start_stop2, y_start_stop3 = [390, 645], [390, 600], [390, 550]

        windows1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop1,
                            xy_window=(128, 128), xy_overlap=(0.75, 0.75))
        windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop2,
                            xy_window=(96, 96), xy_overlap=(0.75, 0.75))
        windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop3,
                            xy_window=(64, 64), xy_overlap=(0.75, 0.75))

        windows = windows1 + windows2 + windows3
        t = time.time()
        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        cv2.imshow('sdf',window_img)
        cv2.imwrite('window_img.jpg', window_img)
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        mpimg.imsave("out.png", heatmap)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')

        cv2.imshow('draw_img', draw_img)
        cv2.imwrite('draw_img.jpg', draw_img)
        cv2.waitKey(0)

    elif input_type == 'video':
        cap = cv2.VideoCapture(input_name)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 28.0, (1280, 720))
        print(input_name)
        while cap.isOpened():

            ret, frame = cap.read()
            if ret==True:

                draw_image = np.copy(frame)
                y_start_stop1, y_start_stop2, y_start_stop3 = [390, 645], [390, 600], [390, 550]

                windows1 = slide_window(frame, x_start_stop=[None, None], y_start_stop=y_start_stop1,
                                        xy_window=(128, 128), xy_overlap=(0.75, 0.75))
                windows2 = slide_window(frame, x_start_stop=[None, None], y_start_stop=y_start_stop2,
                                        xy_window=(96, 96), xy_overlap=(0.75, 0.75))
                windows3 = slide_window(frame, x_start_stop=[None, None], y_start_stop=y_start_stop3,
                                        xy_window=(64, 64), xy_overlap=(0.75, 0.75))

                windows = windows1 + windows2 + windows3

                hot_windows = search_windows(frame, windows, svc, X_scaler, color_space=color_space,
                                             spatial_size=spatial_size, hist_bins=hist_bins,
                                             orient=orient, pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block,
                                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                                             hist_feat=hist_feat, hog_feat=hog_feat)

                window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
                #cv2.imshow('sdf', window_img)
                heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
                # Add heat to each box in box list
                heat = add_heat(heat, hot_windows)

                # Apply threshold to help remove false positives
                heat = apply_threshold(heat, 2)

                # Visualize the heatmap when displaying
                heatmap = np.clip(heat, 0, 255)

                #mpimg.imsave("out.png", heatmap)

                # Find final boxes from heatmap using label function
                labels = label(heatmap)
                draw_img = draw_labeled_bboxes(np.copy(frame), labels)

                #plt.imshow(heatmap)
                #plt.show()
                cv2.imshow('draw_img', draw_img)
                #out.write(draw_img)

                # out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    cv2.waitKey(0)
                # if cv2.waitKey(1) & 0xFF == ord('r'):
                #    cv2.imwrite('check1.jpg', undist_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

