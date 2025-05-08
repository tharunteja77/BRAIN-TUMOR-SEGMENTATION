import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 as cv
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable

# Register custom loss and metrics
@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

@register_keras_serializable()
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    return (intersection + smooth) / (sum - intersection + smooth)

# Load model
model = tf.keras.models.load_model("my_model.keras", custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef, "iou_coef": iou_coef})

# Preprocessing
def preprocess_image(img):
    img = np.array(img)
    img = cv.resize(img, (256, 256))
    img = img / 255.0
    img = img[np.newaxis, :, :, :]
    return img

def overlay_mask_on_image(original_img, mask_img, alpha=0.5):
    original_img_np = np.array(original_img).astype(np.float32) / 255.0

    if original_img_np.shape[:2] != mask_img.shape[:2]:
        mask_img = cv.resize(mask_img, (original_img_np.shape[1], original_img_np.shape[0]))

    mask = mask_img > 0.5
    overlay = np.zeros_like(original_img_np)
    overlay[mask] = [1, 0, 0]  # Red

    blended = cv.addWeighted(original_img_np, 1 - alpha, overlay, alpha, 0)
    return (blended * 255).astype(np.uint8)


# Streamlit UI
st.title(" Brain Tumor Detection from MRI")
st.write("Upload an MRI scan to detect and visualize tumor segmentation.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_width = 300

    st.image(image, caption="Uploaded MRI", width=img_width)


    if st.button("Predict"):
        with st.spinner("Analyzing MRI..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0, :, :, 0]  # Assuming output shape (1, H, W, 1)

            # st.image(prediction, caption="Predicted Tumor Mask", width=img_width, clamp=True)
            if np.sum(prediction > 0.5) > 0:
                 st.error("Tumor Detected!")
            else:
                 st.success("No Tumor Detected!")

            overlayed = overlay_mask_on_image(image, prediction)
            # st.image(overlayed, caption="Overlayed Mask on MRI", width=img_width)
            # Display all three images side by side
            col2, col3 = st.columns(2)
            
            # with col1:
                # st.image(image, caption="Original MRI", width=300)
            
            with col2:
                st.image(prediction, caption="Predicted Mask", width=300, clamp=True)
            
            with col3:
                st.image(overlayed, caption="Overlayed Image", width=300)

