import cv2
import numpy as np

# 视频路径
VIDEO_PATH = './video/test_person1.mp4'
# 结果视频保存路径
RESULT_PATH = 'result1.mp4'
# 指定多边形的顶点坐标并将其转换为numpy格式
polygon_points = np.array([(850, 500), (1000, 500), (1035, 715), (875, 715)],dtype=np.int32)

if __name__ == '__main__':
    # 打开视频
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print('打开视频失败')
        exit()

    fps=capture.get(cv2.CAP_PROP_FPS)# 获取视频帧率
    frame_width=capture.get(cv2.CAP_PROP_FRAME_WIDTH) # 获取视频宽度
    frame_height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)# 获取视频高度
    #
    capture.get(3)# 1280获取宽度 capture.get(cv2.CAP_PROP_GIGA_FRAME_SENS_WIDTH)获取传感器宽度
    capture.get(4)# 720获取高度 capture.get(cv2.CAP_PROP_GIGA_FRAME_SENS_WIDTH)# 获取传感器高度
    capture.get(5)  # 获取帧率 capture.get(cv2.CAP_PROP_FPS)
    videoWriter = None


    # 循环读取每一帧
    while True :
        success, frame =capture.read() # 读取视频的每一帧
        if not success:
            print('读取视频帧失败')
            break
        # 画横线
        cv2.line(frame,(0,int(frame_height/2)),(int(frame_width),int(frame_height/2)),
                 (0,0,255),thickness=3)  # BGR 画横线
        # 画多边形坐标为[polygon_points]
        cv2.polylines(frame,[polygon_points],True,(0,0,255),thickness=3)

        # 创建一个与frame一样的全黑图像
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask,[polygon_points],(0,0,255),)

        # 融合两张图像
        frame = cv2.addWeighted(frame,0.7,mask,0.3,0)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v',)
            videoWriter = cv2.VideoWriter(RESULT_PATH,fourcc,fps,(int(frame_width),int(frame_height)),True)
        videoWriter.write(frame)
        cv2.imshow('frame',frame)# 显示每一帧
        cv2.waitKey(1)# 保持窗口的显示时间1ms

    # 释放资源
    capture.release() # 释放capture
    videoWriter.release() # 释放videoWriter
    cv2.destroyAllWindows() # 销毁所有窗口

