import torch
import gradio as gr

model = torch.hub.load(".", "custom", path="runs/train/exp8/weights/best.pt", source="local")

title = "水果检测模型Yolov5测试（基于Gradio）"
desc = "前端可以调节置信度阈值与交并比阈值，针对不同的精度展示不同的效果"
# 置信度阈值，交并比阈值
base_conf, base_iou = 0.25, 0.50


def det_image(img, conf_thres, iou_thres):
    model.conf = conf_thres
    model.iou = iou_thres
    return model(img).render()[0]


gr.Interface(
    inputs=["image", gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
    outputs=["image"],
    fn=det_image,
    title=title,
    description=desc,
    live=True,
    # 例子，可以不加
    # examples=[["../datasets/eye/train/apple_77_jpg.rf.8bc9fbdc03313f422b7840dcb22f4eaf.jpg", base_conf, base_iou],
    #           ["../datasets/eye/train/apple_80_jpg.rf.943dcb4157d0d83d6e906dfb1edfb30e.jpg", 0.3, base_iou]]).launch(share=True)
).launch(share=True)
# 调用摄像头
# gr.Interface(inputs=[gr.Webcam(),gr.Slider(minimum=0,maximum=1,value = base_conf),gr.Slider(minimum=0,maximum=1,value = base_iou)],
#              outputs=["image"],
#              fn=det_image,
#              title=title,
#              description=desc,
#              examples=[["./datasets/images/train/30.jpg",base_conf,base_iou],["./datasets/images/train/60.jpg",0.3,base_iou]]).launch()

# launch(share=True)