# yolov5-streamlit
该版本为yolov5-5.0版本

Deploy [YOLOv5](https://github.com/ultralytics/yolov5/releases/tag/v5.0) detection with [Streamlit](https://github.com/streamlit/streamlit)

博文地址： <https://xugaoxiang.com/2021/08/27/yolov5-streamlit/>

# 线上体验

直接访问 <https://share.streamlit.io/xugaoxiang/yolov5-streamlit/main/main.py>

# 安装依赖

```
# 本地安装的话，请将opencv-python-headless改为opencv-python
pip install -r requirements.txt
```

如果有`GPU`的话，将`torch`替换成`gpu`版本可加速检测

# 运行项目

```
streamlit run main.py
```

**图片检测**

![streamlit yolov5 image detection](data/images/image.png)

**视频检测**

![streamlit yolov5 video detection](data/images/video.png)

**未完成部分**

![streamlit yolov5 video detection](data/images/1699697017642.jpg)
新增按钮或者模块，保留每次检测完成后的记录  点击可以跳转查看之前的视频