from io import StringIO
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from onnx_detect import YOLOV5_ONNX
import subprocess
from torchvision import models, transforms
import numpy as np
import av
import cv2
import time
import threading
from detect import run as detect
from detect import load_model
# from video_detect import video_detect
import os
import sys
import argparse
from PIL import Image
RTC_CONFIGURATION = RTCConfiguration(
    {
      "RTCIceServer": [{
        "urls": ["stun:stun.l.google.com:19302"],
        "username": "pikachu",
        "credential": "1234",
      }]
    }
)

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_pre_video_folder():
    '''
        Returns the latest folder in a data\videos
    '''
    return max(get_subdirs(os.path.join('data', 'videos')), key=os.path.getmtime)

def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

def ffmpeg_cover_mp4(input_file,output_file):
    ffmpeg_command = f"ffmpeg -i {input_file} -vcodec libx265 -preset slow -b:v 2000k -crf 21 -strict -2 {output_file}"
    subprocess.run(ffmpeg_command, shell=True)
    st.video(output_file)

def progress_threads():
    progress_text = "视频推理中，请稍后...."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.2)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

if __name__ == '__main__':

    st.title('YOLOv5 Streamlit App')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default= 0,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()


    source = ("图片检测", "视频检测", "实时检测")
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    select_weights = ("YOLOv5s", "YOLOv5l", "YOLOv5x","Prune_ONNX")
    select_weights_index = st.sidebar.selectbox("选择模型", range(
        len(select_weights)), format_func=lambda x: select_weights[x])
    if select_weights_index == 0:
        opt.weights = "weights/yolov5s.pt"
    elif select_weights_index == 1:
        opt.weights = "weights/yolov5l.pt"
    elif select_weights_index == 3:
        opt.weights = "weights/yolov5l_prune.onnx"
    elif select_weights_index == 2:
        opt.weights = "weights/yolov5x.pt"
    opt.conf_thres = st.sidebar.slider('阈值', 0.0, 1.0, 0.5)
    opt.iou_thres = st.sidebar.slider('置信度', 0.0, 1.0, 0.5)
    print(opt)
    # model = None
    # if st.sidebar.button("加载模型"):
    #     model = load_model(opt)
    # 判断推理类型，加载数据
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                # 显示缩略图
                # st.sidebar.image(uploaded_file)
                # 显示大图
                st.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    elif source_index == 1:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                # 显示大视频
                st.video(uploaded_file)
                # 显示缩略图
                # st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
                cap = cv2.VideoCapture(opt.source)
                if not cap.isOpened():
                    print("Error: Could not open video file.")
                    exit()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_in_seconds = total_frames / fps

        else:
            is_valid = False
    else:
        class VideoProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = Image.fromarray(img)
                img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                opt.source = img
                # r_image = video_detect(handle, opt)
                r_image = handle.infer(img)
                return av.VideoFrame.from_ndarray(r_image)


        from models.experimental import attempt_load

        # handle = attempt_load("weights/yolov5s.pt", map_location="cpu")
        # handle = YOLOV5_ONNX(onnx_path="./weights/yolov5l.onnx")
        handle = YOLOV5_ONNX(onnx_path="./weights/yolov5l_prune.onnx")
        webrtc_ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor
        )
        is_valid = False

    if is_valid:
        print('valid')
        if st.button('开始检测'):
            # if not model:
            #     st.write("模型未加载，需要先加载模型")
            if source_index == 0:
                infer_thread = threading.Thread(target=detect, args=(opt,))
                infer_thread.start()
                # progress_threads()
                infer_thread.join()
                with st.spinner(text='推理完成，图片准备中'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))
                    st.balloons()
            else:
                infer_thread = threading.Thread(target=detect, args=(opt,))
                infer_thread.start()
                progress_threads()
                infer_thread.join()
                with st.spinner(text='推理完成，视频准备中'):
                    for vid in os.listdir(get_detection_folder()):
                        print(get_detection_folder())
                        print(Path(f'{get_detection_folder()}'))
                        print(str(Path(f'{get_detection_folder()}') / vid))
                        input_file = str(Path(f'{get_detection_folder()}') / vid)
                        output_file = str(Path(f'{get_detection_folder()}') / "output.mp4")
                        print(output_file)
                        ffmpeg_cover_mp4(input_file,output_file)
                        # ffmpeg_thread = threading.Thread(target=ffmpeg_cover_mp4, args=(input_file,output_file))
                        # ffmpeg_thread.start()
                    st.balloons()
