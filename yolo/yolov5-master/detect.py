# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import yaml
import numpy as np
import csv
import os
import platform
import sys
from pathlib import Path
from utils.metrics import box_iou

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.metrics import ap_per_class
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
    weights=ROOT / "runs/train/exp9/weights/best.pt",  # model path or triton URL
    source=ROOT / "data/image_fruits",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/fruits_data.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # print(names) {0: 'apple', 1: 'banana', 2: 'orange'}
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 初始化保存预测和真实值的列表
    pred_list = []
    labels_list = []
    seen = 0

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    windows, dt = [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            # print(det)
            pred_list.append(det)
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    labels_list.append([cls, *xyxy])
                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 计算指标
    iou_list, precision_list, recall_list, map50_list = compute_metrics(pred_list, labels_list, names)

    # 输出指标
    print_metrics(iou_list, precision_list, recall_list, map50_list)

    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def compute_metrics(pred_list, labels_list, names):
    iou_list = []
    precision_list = []
    recall_list = []
    map50_list = []

    # 转换函数
    def convert_to_tensor(labels_list):
        tensor_list = []
        for sublist in labels_list:
            tensor_values = [int(tensor_item.item()) for tensor_item in sublist[:]]
            # 创建 PyTorch Tensor，并添加到 tensor_list 中
            tensor_list.append(torch.tensor(tensor_values))

        # 将 tensor_list 转换为一个大的 PyTorch Tensor
        final_tensor = torch.stack(tensor_list)
        return final_tensor

    # 调用函数转换
    labels_list = convert_to_tensor(labels_list)
    # print(labels_list)
    for cls in range(len(names)):
        cls_preds = []
        cls_labels = []
        # target_cls = [0, 1, 2]
        for preds, labels in zip(pred_list, labels_list):
            cls_preds.append(preds[preds[:, 5].long() == cls])
            cls_labels.append(labels[labels[0] == cls])

        # 打印调试信息
        print(f"cls_preds for class {cls}: {cls_preds}")
        print(f"cls_labels for class {cls}: {cls_labels}")

        # IoU
        # if len(cls_preds) > 0 and len(cls_labels) > 0:
        #     iou = box_iou(torch.cat(cls_preds)[:, :4], torch.cat(cls_labels)[:, 1:5])
        #     iou_list.append(iou.mean().item())
        #
        #     print(f"IoU for class {cls}: {iou.mean().item()}")
        #     print(torch.cat(cls_preds)[:, 4])
        #     print(torch.cat(cls_preds)[:, 5].long())
        #     print(torch.cat(cls_labels)[:, 0].long())
        #     # Precision, Recall, mAP50
        #     precision, recall, ap, f1, ap_class = ap_per_class(
        #         torch.cat(cls_preds)[:, 4], torch.cat(cls_preds)[:, 5].long(), torch.cat(cls_labels)[:, 0].long(), torch.cat(target_cls)
        #     )
        #     precision_list.append(precision.mean().item())
        #     recall_list.append(recall.mean().item())
        #     map50_list.append(ap.mean().item())
        #
        #     print(f"Precision for class {cls}: {precision.mean().item()}")
        #     print(f"Recall for class {cls}: {recall.mean().item()}")
        #     print(f"mAP50 for class {cls}: {ap.mean().item()}")
        # IoU
        if cls_preds and cls_labels:
            iou = box_iou(torch.cat(cls_preds)[:, :4], torch.cat(cls_labels)[:, 1:5])
            iou_list.append(iou.mean().item())

            # Precision, Recall, mAP50
            precision, recall, ap50 = compute_precision_recall_ap(cls_preds, cls_labels, iou)
            precision_list.append(precision)
            recall_list.append(recall)
            map50_list.append(ap50)
        else:
            iou_list.append(0)
            precision_list.append(0)
            recall_list.append(0)
            map50_list.append(0)
    # # Calculate overall metrics
    # iou_all = torch.tensor(iou_list).mean().item()
    # precision_all = torch.tensor(precision_list).mean().item()
    # recall_all = torch.tensor(recall_list).mean().item()
    # map50_all = torch.tensor(map50_list).mean().item()

    # 计算总体指标
    iou_all = np.mean(iou_list)
    precision_all = np.mean(precision_list)
    recall_all = np.mean(recall_list)
    map50_all = np.mean(map50_list)

    iou_list.insert(0, iou_all)
    precision_list.insert(0, precision_all)
    recall_list.insert(0, recall_all)
    map50_list.insert(0, map50_all)

    # iou_list = [0.75, 0.80, 0.78, 0.82]
    # precision_list = [0.90, 0.92, 0.89, 0.91]
    # recall_list = [0.85, 0.88, 0.87, 0.89]
    # map50_list = [0.88, 0.91, 0.89, 0.90]

    return iou_list, precision_list, recall_list, map50_list

def compute_precision_recall_ap(preds, labels, iou_matrix):
    """Computes Precision, Recall, and AP50 for a specific class."""
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    npos = len(labels)

    for i in range(len(preds)):
        iou_max = iou_matrix[i].max().item()
        if iou_max > 0.5:
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / npos

    ap50 = np.mean(precision)  # 平均精度

    return precision[-1], recall[-1], ap50

def print_metrics(iou_list, precision_list, recall_list, map50_list):
    """Prints metrics for each class in a table format."""
    classes = ["All", "Apple", "Banana", "Orange"]
    print(f"{'Class':<10}{'IoU':<10}{'Precision':<10}{'Recall':<10}{'mAP50':<10}")
    for cls, iou, precision, recall, map50 in zip(classes, iou_list, precision_list, recall_list, map50_list):
        print(f"{cls:<10}{iou:<10.2f}{precision:<10.2f}{recall:<10.2f}{map50:<10.2f}")

def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp9/weights/best.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/image_fruits", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/fruits_data.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



#
# import argparse
# import yaml
# import numpy as np
# import csv
# import os
# import platform
# import sys
# from pathlib import Path
#
# import torch
#
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#
# from ultralytics.utils.plotting import Annotator, colors, save_one_box
#
# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (
#     LOGGER,
#     Profile,
#     check_file,
#     check_img_size,
#     check_imshow,
#     check_requirements,
#     colorstr,
#     cv2,
#     increment_path,
#     non_max_suppression,
#     print_args,
#     scale_boxes,
#     strip_optimizer,
#     xyxy2xywh,
# )
# from utils.torch_utils import select_device, smart_inference_mode
#
#
# @smart_inference_mode()
# def run(
#     weights=ROOT / "runs/train/exp9/weights/best.pt",  # model path or triton URL
#     source=ROOT / "data/image_fruits",  # file/dir/URL/glob/screen/0(webcam)
#     data=ROOT / "data/fruits_data.yaml",  # dataset.yaml path
#     imgsz=(640, 640),  # inference size (height, width)
#     conf_thres=0.25,  # confidence threshold
#     iou_thres=0.45,  # NMS IOU threshold
#     max_det=1000,  # maximum detections per image
#     device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#     view_img=False,  # show results
#     save_txt=False,  # save results to *.txt
#     save_csv=False,  # save results in CSV format
#     save_conf=False,  # save confidences in --save-txt labels
#     save_crop=False,  # save cropped prediction boxes
#     nosave=False,  # do not save images/videos
#     classes=None,  # filter by class: --class 0, or --class 0 2 3
#     agnostic_nms=False,  # class-agnostic NMS
#     augment=False,  # augmented inference
#     visualize=False,  # visualize features
#     update=False,  # update all models
#     project=ROOT / "runs/detect",  # save results to project/name
#     name="exp",  # save results to project/name
#     exist_ok=False,  # existing project/name ok, do not increment
#     line_thickness=3,  # bounding box thickness (pixels)
#     hide_labels=False,  # hide labels
#     hide_conf=False,  # hide confidences
#     half=False,  # use FP16 half-precision inference
#     dnn=False,  # use OpenCV DNN for ONNX inference
#     vid_stride=1,  # video frame-rate stride
# ):
#     source = str(source)
#     save_img = not nosave and not source.endswith(".txt")  # save inference images
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
#     webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
#     screenshot = source.lower().startswith("screen")
#     if is_url and is_file:
#         source = check_file(source)  # download
#
#     # Directories
#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
#
#     # Load model
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)  # check image size
#
#     # Dataloader
#     bs = 1  # batch_size
#     if webcam:
#         view_img = check_imshow(warn=True)
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#         bs = len(dataset)
#     elif screenshot:
#         dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#     vid_path, vid_writer = [None] * bs, [None] * bs
#
#     # Run inference
#     model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
#     seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
#     for path, im, im0s, vid_cap, s in dataset:
#         with dt[0]:
#             im = torch.from_numpy(im).to(model.device)
#             im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#             im /= 255  # 0 - 255 to 0.0 - 1.0
#             if len(im.shape) == 3:
#                 im = im[None]  # expand for batch dim
#             if model.xml and im.shape[0] > 1:
#                 ims = torch.chunk(im, im.shape[0], 0)
#
#         # Inference
#         with dt[1]:
#             visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             if model.xml and im.shape[0] > 1:
#                 pred = None
#                 for image in ims:
#                     if pred is None:
#                         pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
#                     else:
#                         pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
#                 pred = [pred, None]
#             else:
#                 pred = model(im, augment=augment, visualize=visualize)
#         # NMS
#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#
#         # Second-stage classifier (optional)
#         # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
#
#         # Define the path for the CSV file
#         csv_path = save_dir / "predictions.csv"
#
#         # Create or append to the CSV file
#         def write_to_csv(image_name, prediction, confidence):
#             """Writes prediction data for an image to a CSV file, appending if the file exists."""
#             data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
#             with open(csv_path, mode="a", newline="") as f:
#                 writer = csv.DictWriter(f, fieldnames=data.keys())
#                 if not csv_path.is_file():
#                     writer.writeheader()
#                 writer.writerow(data)
#
#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f"{i}: "
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
#
#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # im.jpg
#             txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
#             s += "%gx%g " % im.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if save_crop else im0  # for save_crop
#             annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, 5].unique():
#                     n = (det[:, 5] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
#
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     c = int(cls)  # integer class
#                     label = names[c] if hide_conf else f"{names[c]}"
#                     confidence = float(conf)
#                     confidence_str = f"{confidence:.2f}"
#
#                     if save_csv:
#                         write_to_csv(p.name, label, confidence_str)
#
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                         with open(f"{txt_path}.txt", "a") as f:
#                             f.write(("%g " * len(line)).rstrip() % line + "\n")
#
#                     if save_img or save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
#                         annotator.box_label(xyxy, label, color=colors(c, True))
#                     if save_crop:
#                         save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
#
#             # Stream results
#             im0 = annotator.result()
#             if view_img:
#                 if platform.system() == "Linux" and p not in windows:
#                     windows.append(p)
#                     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
#                     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond
#
#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == "image":
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path[i] != save_path:  # new video
#                         vid_path[i] = save_path
#                         if isinstance(vid_writer[i], cv2.VideoWriter):
#                             vid_writer[i].release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                         save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
#                         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#                     vid_writer[i].write(im0)
#
#
#         # Print time (inference-only)
#         LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
#
#     t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
#     LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#     if update:
#         strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
#
#
# def parse_opt():
#     """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp9/weights/best.pt", help="model path or triton URL")
#     parser.add_argument("--source", type=str, default=ROOT / "data/image_fruits", help="file/dir/URL/glob/screen/0(webcam)")
#     parser.add_argument("--data", type=str, default=ROOT / "data/fruits_data.yaml", help="(optional) dataset.yaml path")
#     parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
#     parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
#     parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
#     parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
#     parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
#     parser.add_argument("--view-img", action="store_true", help="show results")
#     parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
#     parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
#     parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
#     parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
#     parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
#     parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
#     parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
#     parser.add_argument("--augment", action="store_true", help="augmented inference")
#     parser.add_argument("--visualize", action="store_true", help="visualize features")
#     parser.add_argument("--update", action="store_true", help="update all models")
#     parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
#     parser.add_argument("--name", default="exp", help="save results to project/name")
#     parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
#     parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
#     parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
#     parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
#     parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
#     parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
#     parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(vars(opt))
#     return opt
#
#
# def main(opt):
#     """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
#     check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
#     run(**vars(opt))
#
#
# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
