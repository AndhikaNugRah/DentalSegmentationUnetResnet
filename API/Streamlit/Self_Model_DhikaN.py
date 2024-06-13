#1.Import Necessary Libraries
import cv2
import streamlit as st  # Framework for building web apps
import PIL  # Python Imaging Library for image processing
from PIL import Image  # Import Image class specifically
import numpy as np  # For numerical operations

from pathlib import Path  # For handling file paths
import torch  # PyTorch library for deep learning
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights  # Object detection model
from torchvision.utils import draw_bounding_boxes  # To draw bounding boxes on images

#2. Load Local Modules:
#These modules contain settings and helper functions
import settings
import helper
from settings import SEGMENTATION_MODEL

#3. Create Containers for Content:
main_container = st.container()  # Container for main app content
container = st.sidebar.container()  # Container for sidebar elements
container.empty()  # Clear the sidebar container

#4. Title and Image Upload:
with main_container:
    st.title("Dental Segmentation by Andhika Nugraha :student:")
    upload = st.file_uploader(label="Upload Your Radiographs & Detect Here :", type=["png", "jpg", "jpeg"])

#5.Sidebar Elements
st.sidebar.header("Image Config")

# Task selection
selected_task = st.sidebar.radio(
    "Select Task",
    ['Detection', 'Segmentation'],
)

import torch

def segmentation_mask(prediction, num_classes, threshold=0.5):
    # Threshold the prediction tensor
    binary_mask = torch.where(prediction > threshold, 1, 0)
    
    # Generate the segmentation mask
    segmentation_mask = torch.zeros((prediction.shape[0], num_classes, prediction.shape[2], prediction.shape[3]))
    for i in range(num_classes):
        segmentation_mask[:, i, :, :] = binary_mask[:, i, :, :] * i
    
    return segmentation_mask

def postprocess(preds, im, im0s):
    if preds is not None and len(preds) > 0:
        for i, pred in enumerate(preds):
            if pred.shape[0] > 0 and pred.shape[1] > 4:
                pred = pred[pred[:, 4] > conf_thres]  # filter out predictions with low confidence
                if pred.shape[0] > 0:
                    if pred.shape[1] > 4:
                        mi = pred.shape[1] - 5  # number of outputs
                        if mi > 0:
                            segmentation_mask = segmentation_mask(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic=agnostic)
                            preds[i] = segmentation_mask
                        else:
                            preds[i] = []
                    else:
                        preds[i] = []
                else:
                    preds[i] = []
            else:
                preds[i] = []
    else:
        preds = []
    return preds

def non_max_suppression(prediction, conf_thres, iou_thres, classes, agnostic):
    if prediction.shape[0] == 0:
        return []
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

#6.Load Pre Trained Model
# Select model path based on task
if selected_task == 'Detection':
    path_model = Path(settings.DETECTION_MODEL)
elif selected_task == 'Segmentation':
    path_model = Path(r'C:\Users\abc\keras_env\Dental_Segmentation_w_api\weights\unet_resnet_model.onnx')

# Load the model with error handling
try:
    model = helper.load_model(path_model)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {path_model}")
    st.error(ex)

#7.Image Processing Logic
if upload and upload.type.endswith(('jpg', 'png', 'jpeg')):
    img = Image.open(upload).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        try:
            if img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = img
                st.image(img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
            default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                 use_column_width=True)
        else:
            if upload and upload.type.endswith(('jpg', 'png', 'jpeg')):
                uploaded_image = img
                # Convert PIL Image to OpenCV image
                img_cv = np.array(uploaded_image)
                # Resize the image to the correct dimensions
                img_cv = cv2.resize(img_cv, (512, 256))
                # Convert the image to a tensor
                img_tensor = torch.tensor(img_cv, dtype=torch.float32).permute(2, 0, 1)  # Permute to BCHW
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        if img_tensor is not None and img_tensor.shape[0] > 0:
            res = model.predict(img_tensor)
            res = postprocess(res, img, img0s)  # Add this line
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                 use_column_width=True)

        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.data)
        except Exception as ex:
            st.write("No image is uploaded yet!")
