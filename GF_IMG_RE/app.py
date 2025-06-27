import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
import streamlit as st
from PIL import Image, ImageFont, ImageDraw
import io, zipfile
import cv2
import numpy as np
from rembg import remove
from ultralytics import YOLO
from typing import Tuple, Optional, Union
import ssl
import warnings
warnings.filterwarnings("ignore")

def main():

    ssl._create_default_https_context = ssl._create_unverified_context

    # ========== Page Configuration ==========
    
    # ========== Model Loading ========== 
    @st.cache_resource
    def load_yolo_model():
        return YOLO("yolov8n-seg.pt")

    model = load_yolo_model()

    # ========== Utility Functions ==========

    def compute_center_of_bbox(bbox: Tuple[int]) -> Tuple[int, int]:
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def enhanced_subject_detection(model, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        First try YOLOv8 segmentation; if that fails, fall back to rembg mask bounding box.
        Returns (x0, y0, x1, y1) or None.
        """
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model.predict(img_cv, classes=0, verbose=False)
        for r in results:
            if r.masks is not None:
                masks = r.masks.xy
                if len(masks) > 0:
                    largest_mask = max(masks, key=lambda m: cv2.contourArea(m))
                    x, y, w, h = cv2.boundingRect(largest_mask.astype(np.int32))
                    return (x, y, x + w, y + h)

        # Fallback: remove background and get alpha mask bbox
        bg_removed = remove(img, post_process_mask=True)
        alpha = bg_removed.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            dx = int((bbox[2] - bbox[0]) * 0.05)
            dy = int((bbox[3] - bbox[1]) * 0.05)
            x0 = max(0, bbox[0] - dx)
            y0 = max(0, bbox[1] - dy)
            x1 = min(img.width, bbox[2] + dx)
            y1 = min(img.height, bbox[3] + dy)
            return (x0, y0, x1, y1)

        return None

    def smart_crop_with_headspace(
        img: Image.Image,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
        zoom: float,
        headspace_top: Union[int, float],
        headspace_bottom: Union[int, float],
        use_percent: bool = False
    ) -> Image.Image:
        """
        Crop around the subject bounding box with optional headspace above/below,
        then resize exactly to target_size.
        """
        img_w, img_h = img.size
        t_w, t_h = target_size

        subject_w = bbox[2] - bbox[0]
        subject_h = bbox[3] - bbox[1]

        zoomed_w = subject_w * zoom
        zoomed_h = subject_h * zoom

        scale = min(t_w / zoomed_w, t_h / zoomed_h)
        crop_w = int(t_w / scale)
        crop_h = int(t_h / scale)

        top_extra = int((headspace_top * subject_h) / 100) if use_percent else int(headspace_top)
        bottom_extra = int((headspace_bottom * subject_h) / 100) if use_percent else int(headspace_bottom)

        cx, cy = compute_center_of_bbox(bbox)
        top = max(0, cy - crop_h // 2 - top_extra)
        bottom = min(img_h, cy + crop_h // 2 + bottom_extra)
        left = max(0, cx - crop_w // 2)
        right = min(img_w, cx + crop_w // 2)

        # Adjust if near edges
        if right - left < crop_w:
            if left == 0:
                right = min(crop_w, img_w)
            else:
                left = max(0, right - crop_w)
        if bottom - top < crop_h:
            if top == 0:
                bottom = min(crop_h, img_h)
            else:
                top = max(0, bottom - crop_h)

        cropped = img.crop((left, top, right, bottom))
        return cropped.resize(target_size, Image.LANCZOS)

    def optimize_image(img: Image.Image, max_size_kb: int) -> io.BytesIO:
        """
        Iteratively reduce JPEG quality until under max_size_kb. Returns a BytesIO.
        """
        buffer = io.BytesIO()
        quality = 95
        img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
        while (buffer.tell() / 1024) > max_size_kb and quality > 10:
            buffer.seek(0)
            buffer.truncate()
            quality -= 5
            img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
        buffer.seek(0)
        return buffer

    def apply_branding(
        img: Image.Image,
        logo: Optional[Image.Image],
        **kwargs
    ) -> Image.Image:
        """
        Apply padding, logo, and optional text overlay.
        """
        composite = img.convert("RGBA")

        # 1) Padding
        if kwargs.get("add_padding", False):
            pad = kwargs.get("padding", 0)
            new_w = composite.width + 2 * pad
            new_h = composite.height + 2 * pad
            color = kwargs.get("padding_color", (255, 255, 255, 0))
            base = Image.new("RGBA", (new_w, new_h), color)
            base.paste(composite, (pad, pad))
            composite = base

        # 2) Logo
        if logo is not None:
            logo = logo.convert("RGBA")
            logo_w = int((kwargs["logo_scale"] / 100) * composite.width)
            logo_h = int(logo_w * (logo.height / logo.width))
            logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
            x_px = int((kwargs["x_offset"] / 100) * (composite.width - logo_w))
            y_px = int((kwargs["y_offset"] / 100) * (composite.height - logo_h))
            composite.paste(logo_resized, (x_px, y_px), logo_resized)

        # 3) Text
        if kwargs.get("add_text", False) and kwargs.get("text", ""):
            draw = ImageDraw.Draw(composite)
            try:
                font = ImageFont.truetype("arial.ttf", kwargs["font_size"])
            except:
                font = ImageFont.load_default()
            tx = int((kwargs["text_x"] / 100) * composite.width)
            ty = int((kwargs["text_y"] / 100) * composite.height)
            draw.text((tx, ty), kwargs["text"], fill=kwargs["text_color"], font=font)

        return composite.convert("RGB")

    def preprocess_uploaded_image(img: Image.Image, max_dim: int = 2048) -> Image.Image:
        """
        Resize any side to max_dim if needed, to keep memory usage sane.
        """
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        return img.convert("RGB")

    # ========== Session State & Mode Selection ==========

    # 1) Keep a dynamic-key so we can CLEAR the uploader when needed:
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    # 2) Sidebar: Clear button
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear Uploaded Files"):
            st.session_state.upload_key += 1
            st.rerun()

    # ========== Common Upload Section ==========

    st.title("Snipster - Resizing Tool")
    st.info("Upload images to start smart cropping and branding.", icon="🛠️")

    uploaded_files = st.file_uploader(
        "📸 Upload Image(s) (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_key}"
    )

    # Convert UploadedFile to PIL so we can preview quickly
    def load_image_from_uploaded(upl) -> Image.Image:
        return Image.open(upl).convert("RGB")

    if uploaded_files:
        st.subheader("🔍 Uploaded Image Preview")
        cols = st.columns(min(4, len(uploaded_files)))
        for idx, upl in enumerate(uploaded_files):
            img = preprocess_uploaded_image(load_image_from_uploaded(upl))
            cols[idx % len(cols)].image(img, use_container_width=True, caption=upl.name)

    # ========== Smart Cropper + Brand Generator ==========

    st.sidebar.markdown("## ✂️ Smart Crop Settings")
    with st.sidebar.expander("📐 Output Dimensions"):
        target_width = st.number_input("Width", 512, 4096, 1200, step=100)
        target_height = st.number_input("Height", 512, 4096, 1800, step=100)
        zoom_factor = st.slider("Zoom Level", 0.5, 3.0, 1.2, 0.1)
        st.markdown("---")
        max_size_kb = st.number_input("Max File Size (KB)", 100, 5000, 800, step=50)

    with st.sidebar.expander("🧠 Headspace & Cropping"):
        use_percent = st.checkbox("Use Percent for Headspace")
        top_space = st.number_input("Top Headspace", 0, 1000, 10)
        bottom_space = st.number_input("Bottom Headspace", 0, 1000, 10)

    st.sidebar.markdown("## 🎨 Branding Options")
    with st.sidebar.expander("🏷️ Logo Settings"):
        logo_file = st.file_uploader("Upload Logo (PNG)", type=["png"])
        logo_scale = st.slider("Logo Size (% of width)", 5, 50, 25)
        x_offset = st.slider("Logo Horizontal Pos (%)", 0, 100, 50)
        y_offset = st.slider("Logo Vertical Pos (%)", 0, 100, 90)

    with st.sidebar.expander("🔤 Text Overlay"):
        add_text = st.checkbox("Add Text")
        if add_text:
            text = st.text_input("Text Content", "Your Brand Message")
            font_size = st.slider("Font Size", 10, 150, 90)
            text_color = st.color_picker("Text Color", "#000000")
            text_x = st.slider("Text Horizontal Pos (%)", 0, 100, 50)
            text_y = st.slider("Text Vertical Pos (%)", 0, 100, 90)
        else:
            text = ""
            font_size = 40
            text_color = "#000000"
            text_x = 5
            text_y = 5

    with st.sidebar.expander("🧱 Padding"):
        add_padding = st.checkbox("Add Padding")
        if add_padding:
            padding = st.slider("Padding (px)", 0, 300, 50)
            padding_color = st.color_picker("Padding Color", "#FFFFFF")
        else:
            padding = 0
            padding_color = "#FFFFFF"
            add_padding = False

    # Only show "Process" if there is at least one upload
    if uploaded_files:
        if st.button("🚀 Process Images"):
            results = []
            logo_img = None
            if logo_file:
                logo_img = Image.open(logo_file).convert("RGBA")

            progress = st.progress(0, text="Processing…")
            for i, upl in enumerate(uploaded_files):
                base_img = preprocess_uploaded_image(load_image_from_uploaded(upl))
                if max(base_img.size) > 3000:
                    base_img = base_img.resize((base_img.width // 2, base_img.height // 2), Image.LANCZOS)

                bbox = enhanced_subject_detection(model, base_img)
                if not bbox:
                    # fallback to center if detection fails
                    w, h = base_img.size
                    bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

                cropped = smart_crop_with_headspace(
                    base_img, bbox, (target_width, target_height),
                    zoom_factor, top_space, bottom_space, use_percent
                )

                branded_img = apply_branding(
                    cropped, logo_img,
                    logo_scale=logo_scale, x_offset=x_offset, y_offset=y_offset,
                    add_text=add_text, text=text, font_size=font_size,
                    text_color=text_color, text_x=text_x, text_y=text_y,
                    add_padding=add_padding, padding=padding, padding_color=padding_color
                )

                buf = optimize_image(branded_img, max_size_kb)
                results.append((upl.name, branded_img, buf))
                progress.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1}/{len(uploaded_files)}")

            progress.empty()

            st.subheader("🎨 Branded Output Preview")
            preview_cols = st.columns(min(4, len(results)))
            for idx, (fname, img_obj, buff) in enumerate(results):
                with preview_cols[idx % len(preview_cols)]:
                    st.image(img_obj, caption=fname, use_container_width=True)
                    st.download_button(
                        label="⬇️ Download",
                        data=buff.getvalue(),
                        file_name=f"branded_{fname}",
                        mime="image/jpeg",
                        key=f"download_{idx}"
                    )

            # ZIP all
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for fname, _, buff in results:
                    zf.writestr(f"branded_{fname}", buff.getvalue())
            zip_buf.seek(0)
            st.download_button(
                "📦 Download All as ZIP",
                data=zip_buf.getvalue(),
                file_name="branded_images.zip",
                mime="application/zip"
            )

    else:
        st.info("Upload at least one image to start cropping + branding.")