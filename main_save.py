#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0                                           NAM MÔ A DI ĐÀ PHẬT
#                    ___/`---'\___                  Thí chủ con tên là Thân Ngọc Thiện, dương lịch hai sáu tháng ba năm 2003,
#                  .' \\|     |// '.            Hiện tạm trú tại Phú Diễn, Hà Nội. Nguyên quán: Nghĩa Trung, TX. Việt Yên, Bắc Giang
#                 / \\|||  :  |||// \           
#                / _||||| -:- |||||- \                      Con lạy chín phương trời, con lạy mười phương đất
#               |   | \\\  -  /// |   |                         Chư Phật mười phương, mười phương chư Phật
#               | \_|  ''\---/''  |_/ |                         Cảm ơn trời đất trở che, thần thánh cứu độ
#               \  .-\__  '-'  ___/-. /                Xin nhất tâm kính lễ Hoàng thiên Hậu thổ, Tiên Phật Thánh Thần
#             ___'. .'  /--.--\  `. .'___                             Giúp đỡ con code sạch ít bugs
#          ."" '<  `.___\_<|>_/___.' >' "".                        Đồng nhiệp vui vẻ, sếp quý, lương cao
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |                       Sức khỏe dồi dào, tiền vào như nước
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            Phật phù hộ, không bao giờ Bug
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import cv2
import time
import serial
import numpy as np
from datetime import datetime

from yolov8 import YOLOv8, draw_detections


yolov8_detector_ball = YOLOv8('weights/best_e20.onnx', conf_thres=0.5, iou_thres=0.5)
yolov8_detector_silo = YOLOv8('weights/best_e20.onnx', conf_thres=0.5, iou_thres=0.5)

STATE = '0'
STATE_DETECT_SILO = '0'
STATE_DETECT_BALL = '1'
SILO_SELECTED_1 = 0
SILO_SELECTED_2 = 0

# Initialize the webcam
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

if not os.path.exists('./save-frame/'):
    os.mkdir('./save-frame/')
out_name = './save-frame/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
fps = 30.0
origin_out = cv2.VideoWriter(f'{out_name}_origin.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap0.get(3)), int(cap0.get(4))))
visulize_out = cv2.VideoWriter(f'{out_name}_visulize.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap0.get(3)), int(cap0.get(4))))
NUM_FRAME_SAVED = 0

def export_video(org_img, vis_img):
    global NUM_FRAME_SAVED

    origin_out.write(org_img)
    cv2.putText(vis_img, str(NUM_FRAME_SAVED), (0, CAM_Y_CEN*2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    visulize_out.write(vis_img)
    NUM_FRAME_SAVED += 1
    print(f"Added {NUM_FRAME_SAVED}th frame to {out_name}_*.mp4")

def get_state():
    if ser.inWaiting() == 0:
            #print(0)
            return STATE

    # Đoạn này trở đi là nhan lenh tu serial
    data1 = ser.readline(1)
    print(data1)
    try:
        data1 = str(data1, 'utf-8')
    except UnicodeDecodeError:
        # Xử lý khi gặp lỗi UnicodeDecodeError
        data1 = str(data1, 'latin-1')  # hoặc mã hóa khác
    print(data1.strip('\r\n'))
    return data1.strip('\r\n')

def get_frame():
    global STATE
    current_state = get_state()
    if current_state != STATE and current_state != '':
        STATE = current_state
    if STATE == STATE_DETECT_BALL:
        return cap0.read()
    else:
        return cap1.read()

def detect_ball(frame):
    start_time = time.time()
    boxes, scores, class_ids = yolov8_detector_ball(frame)
    print(f'{(time.time() - start_time)*1000:.2f} ms')
    
    return boxes, scores, class_ids

def detect_silo(frame):
    start_time = time.time()
    boxes, scores, class_ids = yolov8_detector_silo(frame)
    print(f'{(time.time() - start_time)*1000:.2f} ms')

    # # xoa o day
    # idxs = class_ids == 2
    # for i in range(3, 9):
    #     idxs = np.logical_or(idxs, class_ids == i)
    # boxes, scores, class_ids = boxes[idxs], scores[idxs], class_ids[idxs]
    # # den day

    if len(boxes) <= 1:
        return boxes, scores, class_ids

    idxs = np.argsort(boxes[:,0])
    boxes, scores, class_ids = boxes[idxs], scores[idxs], class_ids[idxs]

    idx_keep = [True]*len(boxes)
    for i in range(len(boxes)):
        if idx_keep[i]:
            for j in range(i+1, len(boxes)):
                if abs(boxes[i][0]-boxes[j][0]) <= 10:
                    idx_keep[(i, j)[scores[i]>scores[j]]] = False

    boxes, scores, class_ids = boxes[idx_keep], scores[idx_keep], class_ids[idx_keep]

    if len(boxes) > 5:
        idx_get = np.argsort(class_ids)[:5]

        # return boxes, scores, class_ids
        return boxes[idx_get], scores[idx_get], class_ids[idx_get]
    else:
        return boxes, scores, class_ids+2

#Hàm lọc box theo id tương ứng: 
def filter_boxes(boxes, scores, class_ids, id):
    idxs = class_ids == id
    return boxes[idxs], scores[idxs], class_ids[idxs]

def get_box_id(boxes):
    '''
        sort silo/ball by area, if area isn't good for choice object, using distance with center frame
    '''
    x_c, y_c = (boxes[:,2] + boxes[:,0])/2, (boxes[:,3] + boxes[:,1])/2
    id_get = np.argmin(np.sqrt((x_c-CAM_X_CEN)**2 + (y_c-CAM_Y_CEN)**2))
    #CURRENT_ID = ids[id_get]
    return id_get

def get_output(data, lenght=8):
    if STATE == STATE_DETECT_BALL:
        data = (lenght - len(data.split(';')))*'0;' + data +'\r'
    else:
        data = ';'.join(data.split(';')[:-2]) + (lenght - len(data.split(';')))*';0' + ';' + ';'.join(data.split(';')[-2:]) +'\r'
    return data

def main():
    while True:
        # Read frame from the video
        ret, frame = get_frame()

        assert ret, 'Camera error!!!'

        #Phân loại từng loại tương ứng với từng trường hợp trong san xanh 
        if STATE == STATE_DETECT_BALL:
            boxes, scores, class_ids = detect_ball(frame)

            visualize_img = draw_detections(frame, boxes, scores, class_ids)

            boxes_filter, _, _ = filter_boxes(boxes, scores, class_ids, SELECED_BALL)

            if len(boxes_filter) > 0:
                box = boxes[get_box_id(boxes_filter)]
                x, y = int(box[2] + box[0])//2, int(box[3] + box[1])//2
            else:
                x, y = 0, 0
            data = get_output(str(x)+';'+str(y))
            ser.write(data.encode())
            visualize_img = cv2.circle(visualize_img, (x, y), 13, (255, 255, 255) , -1)
        else:
            boxes, scores, class_ids = detect_silo(frame)

            visualize_img = draw_detections(frame, boxes, scores, class_ids)
            
            # if len(boxes) > 0:
            #     print(class_ids[np.argsort(boxes[:,0])])
            #     print(scores[np.argsort(boxes[:,0])])

            freq = {l:c for l, c in zip(*np.unique(class_ids, return_counts=True))}
            choiced_id = []
            for i in ORDER_TO_GET_SILO:
                if i in freq.keys():
                    choiced_id.append(i)
            
            boxes_filters = []
            for i in choiced_id:
                boxes_filter, _, _ = filter_boxes(boxes, scores, class_ids, i)
                boxes_filters.append(boxes_filter)

            if len(boxes_filters) > 0:  
                boxes_filters = np.concatenate(boxes_filters, axis=0)           
                x, y = int(boxes_filters[0][2] + boxes_filters[0][0])//2, int(boxes_filters[0][3] + boxes_filters[0][1])//2
                order_silo_selected = [str(it+1) for it in np.argsort(boxes_filters[:,0])]
                for idx, box in enumerate(boxes_filters):
                    x_c, y_c = int(box[2] + box[0])//2, int(box[3] + box[1])//2
                    cv2.putText(visualize_img, str(idx), (x_c, y_c), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            else:
                order_silo_selected = []
                x, y = 0, 0
            
            data = get_output(f'{len(boxes)};{";".join(order_silo_selected)};{x};{y}')
            print(data)
            ser.write(data.encode())

        cv2.imshow("Detected Objects", visualize_img)
        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        export_video(frame, visualize_img)

    cap0.release() 
    cap1.release() 
    origin_out.release() 
    visulize_out.release() 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--blue_yard', action="store_true", help='')
    args = parser.parse_args()

    '''    
    Quy ước: 
    ['ball_b', 'ball_r', 'silo0', 'silo1_b', 'silo1_r', 'silo_bb', 'silo2_rb_br', 'silo2_rr', 'silo3']
    [  '!',       '!',      '!',      '!',      '!',      '!',         '!',          '!',       '!'  ]
    [  '0',       '1',      '2',      '3',      '4',      '5', 	       '6', 	     '7',       '8'  ]
    '''

    if args.blue_yard:
        # blue yard
        SELECED_BALL = 0
        ORDER_TO_GET_SILO = [6, 5, 7, 3, 2, 4]
    else:
        # red yard
        SELECED_BALL = 1
        ORDER_TO_GET_SILO = [6, 7, 5, 4, 2, 3]
    
    while True:
        try:
            ser = serial.Serial('/dev/ttyUSB0', 115200)

            CAM_X_CEN, CAM_Y_CEN = int(cap0.get(3))//2, int(cap0.get(4)) - 1

            main()
            
            break
        except Exception as e:
            print(e)
