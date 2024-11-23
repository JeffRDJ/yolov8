import cv2

from ultralytics import YOLO, solutions
import os

# 避免KMP报错
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 映入模型
model = YOLO('yolov8n.pt')

# 视频路径
VIDEO_PATH = './video/test_traffic1.mp4'
# 结果保存路径
RESULT_PATH = './demo_video/result_traffic3.mp4'
capture = cv2.VideoCapture(VIDEO_PATH)
# 设置断言，若不存在则报错
assert capture.isOpened(), "Error reading file file"

# 获取视频尺寸和帧率
w, h, fps = (int(capture.get(x)) for x in
             (cv2.CAP_PROP_FRAME_WIDTH,
              cv2.CAP_PROP_FRAME_HEIGHT,
              cv2.CAP_PROP_FPS)
             )
# 设置划线的坐标点
line_points = [(0, h // 2), (w, h // 2)]

# 创建视频写入对象
video_writer = cv2.VideoWriter(
    RESULT_PATH,
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
    fps, (w, h), True
)
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2
)
while capture.isOpened():
    success, im0 = capture.read()
    if not success:
        print("视频读取完成")
        break

    # 模型推理 persist 是否持久化,show 是否显示中间结果
    tracks = model.track(im0, persist=True, show=False)

    # 对im0 帧中的tracks 进行计数
    im0 = counter.start_counting(im0, tracks)

    # 持久化
    video_writer.write(im0)

# 销毁所有窗口和用于持久化的对象，避免资源泄露或者程序崩溃
capture.release()
video_writer.release()
cv2.destroyAllWindows()
