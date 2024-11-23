import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = YOLO('yolov8n.pt')
# 视频路径
VIDEO_PATH = './video/test_person1.mp4'
# 结果保存路径
RESULT_PATH = './demo_video/result_person1.mp4'
# 记录所有的id的位置信息
track_history = defaultdict(lambda : [])

if __name__ == '__main__':
    # 打开视频
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print('打开视频失败')
        exit()

    fps = capture.get(cv2.CAP_PROP_FPS)  # 帧率
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    #
    print(capture.get(3))  # 1280获取宽度 capture.get(cv2.CAP_PROP_GIGA_FRAME_SENS_WIDTH)获取传感器宽度
    print(capture.get(4))  # 720获取高度 capture.get(cv2.CAP_PROP_GIGA_FRAME_SENS_WIDTH)# 获取传感器高度
    print(capture.get(5))  # 获取帧率 capture.get(cv2.CAP_PROP_FPS)
    # 定义一个视频写入器
    videoWriter = None

    # 循环读取每一帧
    while True:
        success, frame = capture.read()  # 读取视频的每一帧
        if not success:
            print('读取视频帧失败')
            break
        # 连续检测视频的帧
        results = model.track(frame, persist=True)
        # 可视化显示
        a_frame = results[0].plot()  # 获取检测结果   0类是人
        # 所有id的位置信息
        boxes = results[0].boxes.xywh.cpu()
        # 所有id的序列号信息
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            # 最长50个点

            x, y, w, h = box
            track = track_history[track_id]
            #x,y是tensor类型，需要转换为float
            track.append((float(x),float(y)))
            # 保持50个点
            if len(track) > 50:
                track.pop(0)
            #当前的track_id所有经过的轨迹路径点
            # points = np.array(track).astype(np.int32).reshape(-1,1,2)
            points = np.hstack(track).astype(np.int32).reshape(-1,1,2)
            # 画点
            cv2.polylines(a_frame,[points],False,(0,255,255),thickness=3)
            cv2.imshow('yolo track', a_frame)
        # 保存视频
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (int(frame_width), int(frame_height)),True)
        videoWriter.write(a_frame)
        cv2.imshow('yolo track', a_frame)
        cv2.waitKey(1)
    # 释放资源
    capture.release()  # 释放capture
    videoWriter.release() # 释放videoWriter
    cv2.destroyAllWindows()  # 销毁所有窗口
