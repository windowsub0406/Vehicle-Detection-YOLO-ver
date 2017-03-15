from yolo_small import *
from calibration import calib, undistort
from threshold import gradient_combine, hls_combine, comb_result
from finding_lines import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map

input_type = 'video' #'video' # 'image'
input_name = 'project_video.mp4' #'test_images/test1.jpg' # 'project_video.mp4'

yolo = yolo_tf()
left_line = Line()
right_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()

if __name__ == '__main__':

    if input_type == 'image':
        frame = cv2.imread(input_name)
        detect_from_file(yolo, frame)

        yolo_result = show_results(frame, yolo)
        cv2.imshow('result', yolo_result)
        cv2.waitKey(0)

    elif input_type == 'video':
        cap = cv2.VideoCapture(input_name)
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('output.avi', fourcc, 25.0, (640, 360))
        while cap.isOpened():
            ret, frame = cap.read()

            if ret == True:
                # Correcting for Distortion
                undist_img = undistort(frame, mtx, dist)
                # resize video
                undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
                detect_from_file(yolo, undist_img)

                yolo_result = show_results(undist_img, yolo)
                #cv2.imshow('result', yolo_result)

                rows, cols = undist_img.shape[:2]
                combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
                combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
                combined_result = comb_result(combined_gradient, combined_hls)

                c_rows, c_cols = combined_result.shape[:2]
                s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
                s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

                src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
                dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

                warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
                searching_img = find_LR_lines(warp_img, left_line, right_line)
                w_color_result = draw_lane(searching_img, left_line, right_line)

                # Drawing the lines back down onto the road
                color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
                lane_color = np.zeros_like(undist_img)
                lane_color[220:rows - 12, 0:cols] = color_result

                # Combine the result with the original image
                result = cv2.addWeighted(yolo_result, 1, lane_color, 0.3, 0)

                info, info2 = np.zeros_like(result), np.zeros_like(result)
                info[5:110, 5:190] = (255, 255, 255)
                info = cv2.addWeighted(result, 1, info, 0.2, 0)
                info2 = cv2.addWeighted(info, 1, info2, 0.2, 0)
                info2 = print_road_status(info2, left_line, right_line)

                cv2.imshow('result', info2)
                #out.write(info2)
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    cv2.imwrite('check1.jpg', info2)
                #cv2.waitKey(0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        #out.release()
        cv2.destroyAllWindows()
