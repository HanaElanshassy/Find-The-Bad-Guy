import streamlit as st
import pandas as pd
from PIL import Image
import os
import cv2
import numpy as np
import base64
import io
import matplotlib.pyplot as plt

# =============================================
# CUSTOM THEME & STYLING
# =============================================

def set_custom_theme():
    """Set custom theme with blue neon palette"""
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #00d4ff;
            --secondary-color: #0080ff;
            --dark-color: #001a33;
            --light-color: #e6f7ff;
            --accent-color: #00ffff;
        }
        
        .stApp {
            background-color: var(--dark-color);
            color: var(--light-color);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color) !important;
            text-shadow: 0 0 10px var(--primary-color);
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .image-card {
            border: 2px solid var(--primary-color);
            border-radius: 10px;
            padding: 10px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .image-card:hover {
            box-shadow: 0 0 15px var(--primary-color);
            transform: scale(1.05);
        }
        
        .image-card.selected {
            border-color: var(--accent-color);
            box-shadow: 0 0 20px var(--accent-color);
        }
        
        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 5px;
        }
        
        .image-card .caption {
            text-align: center;
            margin-top: 8px;
            font-size: 0.8rem;
            color: var(--light-color);
        }
        
        .progress-text {
            color: var(--accent-color);
            font-size: 0.9rem;
            margin-top: 5px;
        }
        
        .load-more-btn {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        /* Enhancement tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px 4px 0 0;
            background-color: var(--dark-color);
            border: 1px solid var(--primary-color);
            transition: all 0.2s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #003366;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color) !important;
            color: var(--dark-color) !important;
        }
        
        /* Slider styling */
        .stSlider [data-testid="stTickBar"] {
            background-color: var(--light-color);
        }
        
        .stSlider [data-testid="stTickBar"] span {
            color: var(--light-color);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================

# Paths to your individual CSV files
CSV_PATHS = {
    "face_shape": r"C:\Users\hanae\Downloads\labeled_faces (1).csv",
    "eye_color": r"C:\Users\hanae\Downloads\labeled_eyes (1).csv",
    "nose_shape": r"C:\Users\hanae\Downloads\labeled_noses (2).csv",
    "skin_tone": r"C:\Users\hanae\Downloads\labeled_skin_tones (1).csv",
    "hair_color": r"C:\Users\hanae\Downloads\labeled_hair (1).csv",
    "skin_marks": r"C:\Users\hanae\Downloads\labeled_skin_marks.csv",
    "mouth_shape": r"C:\Users\hanae\Downloads\labeled_mouth_shapes.csv"
}

# Base directory where your images are stored
IMAGE_BASE_DIR = r"C:\Users\hanae\OneDrive\Desktop\archive (1)"

# =============================================
# DATA LOADING AND MERGING
# =============================================

@st.cache_data
def load_and_merge_data(csv_paths):
    """Load and merge all CSV files"""
    merged_df = None
    for name, path in csv_paths.items():
        try:
            df = pd.read_csv(path)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on="filename", how="outer")
        except Exception as e:
            st.error(f"❌ Error loading {name} from {path}: {e}")
    
    if merged_df is not None:
        merged_df.dropna(inplace=True)
        
    return merged_df

# =============================================
# IMAGE SEARCH FUNCTIONALITY
# =============================================

def find_matching_images(df, selected_attributes, batch_size=10, offset=0):
    """Find matching images with pagination support"""
    if not selected_attributes:
        return [], 0
    
    # Apply all filters to the dataframe
    query_df = df.copy()
    for col, value in selected_attributes.items():
        query_df = query_df[query_df[col] == value]
    
    # Get total matches
    total_matches = len(query_df)
    
    # Process current batch
    matching_images = []
    processed = 0
    skipped = 0
    seen_filenames = set()  # Track seen filenames
    
    while processed < batch_size and (offset + processed + skipped) < total_matches:
        row = query_df.iloc[offset + processed + skipped]
        filename = row['filename']
        
        # Skip if we've already seen this filename
        if filename in seen_filenames:
            skipped += 1
            continue
            
        found = False
        
        # Search for the file
        for root, dirs, files in os.walk(IMAGE_BASE_DIR):
            if filename in files:
                matching_images.append({
                    'path': os.path.join(root, filename),
                    'filename': filename,
                    'attributes': row.to_dict()
                })
                seen_filenames.add(filename)
                found = True
                processed += 1
                break
        
        if not found:
            # Try case insensitive search if exact match fails
            clean_lower = filename.lower()
            for root, dirs, files in os.walk(IMAGE_BASE_DIR):
                for file in files:
                    if file.lower() == clean_lower:
                        matching_images.append({
                            'path': os.path.join(root, file),
                            'filename': file,
                            'attributes': row.to_dict()
                        })
                        seen_filenames.add(file.lower())
                        found = True
                        processed += 1
                        break
                if found:
                    break
        
        if not found:
            skipped += 1
            
    return matching_images, total_matches

# =============================================
# ADVANCED IMAGE ENHANCEMENT
# =============================================

def enhance_image(image_path):
    """Enhanced image enhancement with multiple techniques"""
    st.markdown("---")
    st.markdown("#### 🔮 Advanced Image Enhancement")
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Initialize face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Enhancement functions
        def detect_faces(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return face_cascade.detectMultiScale(gray, 1.3, 5)
        
        def gamma_transform(channel, gamma):
            norm = channel / 255.0
            out = np.power(norm, gamma)
            return np.clip(out * 255, 0, 255).astype(np.uint8)
        
        def gray_level_slicing(img_bgr, min_val, max_val):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask = (gray >= min_val) & (gray <= max_val)
            out = np.zeros_like(img_bgr)
            out[mask] = img_bgr[mask]
            return out
        
        def piecewise_enhancement(img_bgr, contrast, brightness, midpoint, sharpness, use_color_mask, color):
            out = np.zeros_like(img_bgr, dtype=np.float32)
            
            if use_color_mask:
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                if color == 'red':
                    lower1 = np.array([0, 50, 50])
                    upper1 = np.array([10, 255, 255])
                    lower2 = np.array([160, 50, 50])
                    upper2 = np.array([179, 255, 255])
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask1, mask2)
                elif color == 'green':
                    lower = np.array([35, 50, 50])
                    upper = np.array([85, 255, 255])
                    mask = cv2.inRange(hsv, lower, upper)
                img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
            
            for i in range(3):
                channel = img_bgr[..., i].astype(np.float32) / 255.0
                transformed = 1 / (1 + np.exp(-sharpness * (channel - midpoint)))
                transformed = transformed * contrast + (brightness / 255.0)
                out[..., i] = np.clip(transformed * 255, 0, 255)
            
            return out.astype(np.uint8)
        
        def clahe_enhancement(img_bgr, clip_limit, grid_size):
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        def log_transform(img_bgr, c=1.0, base=np.e, channel='All'):
            img = img_bgr.copy().astype(np.float32)
            if channel == 'R':
                img[..., 0] = c * (np.log(img[..., 0] + 1) / np.log(base))
            elif channel == 'G':
                img[..., 1] = c * (np.log(img[..., 1] + 1) / np.log(base))
            elif channel == 'B':
                img[..., 2] = c * (np.log(img[..., 2] + 1) / np.log(base))
            else:
                img = c * (np.log(img + 1) / np.log(base))
            return np.clip(img, 0, 255).astype(np.uint8)
        
        # Create tabs for different enhancement techniques
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Original", 
            "CLAHE Enhanced", 
            "Gamma Corrected", 
            "Gray-level Slicing",
            "Piecewise Enhancement",
            "Log Transform"
        ])
        
        # Get face-only version if requested
        face_only = st.checkbox("Focus on face only", value=False)
        if face_only:
            faces = detect_faces(img_rgb)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                img = img[y:y+h, x:x+w]
                img_rgb = img_rgb[y:y+h, x:x+w]
            else:
                st.warning("No faces detected in the image")
        
        # Original image
        with tab1:
            st.image(img_rgb, use_column_width=True)
            st.markdown("**Original Image**")
        
        # CLAHE Enhancement
        with tab2:
            clahe_col1, clahe_col2 = st.columns(2)
            with clahe_col1:
                clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0, key="clahe_clip")
            with clahe_col2:
                grid_size = st.slider("Grid Size", 2, 16, 8, key="clahe_grid")
            
            enhanced_clahe = clahe_enhancement(img, clip_limit, grid_size)
            st.image(cv2.cvtColor(enhanced_clahe, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown("**CLAHE Enhanced** - Contrast Limited Adaptive Histogram Equalization")
        
        # Gamma Correction
        with tab3:
            gamma = st.slider("Gamma Value", 0.1, 3.0, 1.0, key="gamma_val")
            
            # Apply gamma to each channel
            r, g, b = cv2.split(img)
            R = gamma_transform(r, gamma)
            G = gamma_transform(g, gamma)
            B = gamma_transform(b, gamma)
            enhanced_gamma = cv2.merge((R, G, B))
            
            st.image(cv2.cvtColor(enhanced_gamma, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown(f"**Gamma Correction** (γ = {gamma})")
        
        # Gray-level Slicing
        with tab4:
            min_val, max_val = st.slider(
                "Gray Level Range", 
                0, 255, (100, 160), 
                key="gray_slice"
            )
            
            enhanced_slice = gray_level_slicing(img, min_val, max_val)
            st.image(cv2.cvtColor(enhanced_slice, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown(f"**Gray-level Slicing** ({min_val}-{max_val})")
        
        # Piecewise Enhancement
        with tab5:
            piece_col1, piece_col2 = st.columns(2)
            with piece_col1:
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, key="piece_contrast")
                brightness = st.slider("Brightness", -100.0, 100.0, 0.0, key="piece_brightness")
            with piece_col2:
                midpoint = st.slider("Midpoint", 0.0, 1.0, 0.5, key="piece_midpoint")
                sharpness = st.slider("Sharpness", 1.0, 20.0, 10.0, key="piece_sharpness")
            
            color_mask = st.checkbox("Apply Color Mask", key="piece_mask")
            color = st.selectbox("Mask Color", ["red", "green"], key="piece_color")
            
            enhanced_piece = piecewise_enhancement(
                img, 
                contrast, 
                brightness, 
                midpoint, 
                sharpness,
                color_mask,
                color
            )
            st.image(cv2.cvtColor(enhanced_piece, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown("**Piecewise Enhancement**")
        
        # Log Transform
        with tab6:
            log_col1, log_col2 = st.columns(2)
            with log_col1:
                c = st.slider("Scale (c)", 0.1, 5.0, 1.0, key="log_c")
                base = st.slider("Log Base", 1.1, 10.0, float(np.e), key="log_base")
            with log_col2:
                channel = st.selectbox(
                    "Channel", 
                    ["All", "R", "G", "B"], 
                    key="log_channel"
                )
            
            enhanced_log = log_transform(img, c=c, base=base, channel=channel)
            st.image(cv2.cvtColor(enhanced_log, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown(f"**Log Transform** (c={c}, base={base:.2f}, channel={channel})")
        

        
    except Exception as e:
        st.error(f"Error during enhancement: {e}")

# =============================================
# HELPER FUNCTIONS
# =============================================

def image_to_base64(img):
    """Convert PIL image to base64 for HTML display"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    """Run the main application"""
    set_custom_theme()
    
    # Initialize session state
    if 'page_offset' not in st.session_state:
        st.session_state.page_offset = 0
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'total_matches' not in st.session_state:
        st.session_state.total_matches = 0
    if 'current_matches' not in st.session_state:
        st.session_state.current_matches = []
    
    st.title("🔍 Find The Bad Guy")
    st.markdown("### AI-Powered Facial Attribute Search")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading databases..."):
        df = load_and_merge_data(CSV_PATHS)
    
    if df is None:
        st.error("Failed to load data. Please check your CSV paths.")
        return
    
    attribute_columns = {
        'eye_color': '👁️ Eye Color',
        'hair_color': '💇 Hair Color',
        'mouth_shape_label': '👄 Mouth Shape',
        'nose_shape_label': '👃 Nose Shape',
        'skin_tone_label': '🎨 Skin Tone',
        'face_shape_label': '👤 Face Shape',
        'skin_mark_label': '🔍 Skin Mark'
    }
    
    st.markdown("### 🧠 What do you remember about the person?")
    st.markdown("Select attributes to search our database")
    
    # Create checkboxes and dropdowns
    selected_attributes = {}
    cols = st.columns(2)
    
    for i, (col, label) in enumerate(attribute_columns.items()):
        with cols[i % 2]:
            if st.checkbox(label, key=f"cb_{col}"):
                values = df[col].dropna().unique().tolist()
                selected_value = st.selectbox(
                    label,
                    options=values,
                    key=f"sb_{col}"
                )
                selected_attributes[col] = selected_value
    
    if st.button("🚀 Launch Search", type="primary"):
        # Reset pagination when new search is performed
        st.session_state.page_offset = 0
        st.session_state.selected_image = None
        st.session_state.current_matches = []
        st.session_state.total_matches = 0
        
        # Perform search
        with st.spinner("Searching for matches..."):
            matching_images, total_matches = find_matching_images(
                df, 
                selected_attributes,
                batch_size=10,
                offset=0
            )
            
            st.session_state.current_matches = matching_images
            st.session_state.total_matches = total_matches
    
    # Display current results
    if st.session_state.current_matches:
        st.markdown(f"Showing {len(st.session_state.current_matches)} of {st.session_state.total_matches} matches")
        
        # Display the image grid
        st.markdown("### 🖼️ Matching Images")
        st.markdown("Click on an image to select it for enhancement")
        
        st.markdown('<div class="image-grid">', unsafe_allow_html=True)
        
        cols = st.columns(4)
        col_idx = 0
        for img_info in st.session_state.current_matches:
            try:
                if not os.path.exists(img_info['path']):
                    st.warning(f"Image not found: {img_info['path']}")
                    continue
                
                img = Image.open(img_info['path'])
                
                with cols[col_idx % 4]:
                    is_selected = st.session_state.selected_image == img_info['path']
                    card_class = "image-card selected" if is_selected else "image-card"
                    
                    st.markdown(
                        f'<div class="{card_class}">'
                        f'<img src="data:image/png;base64,{image_to_base64(img)}" alt="{img_info["filename"]}">'
                        f'<div class="caption">{img_info["filename"][:15]}...</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    if st.button(f"Select {col_idx+1}", key=f"select_{col_idx}"):
                        st.session_state.selected_image = img_info['path']
                        st.rerun()
                        
                col_idx += 1
                
            except Exception as e:
                st.error(f"Error loading image {img_info['filename']}: {e}")
                continue

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load more button if there are more matches
        if st.session_state.page_offset + 10 < st.session_state.total_matches:
            if st.button("⬇️ Load More Images", key="load_more", use_container_width=True):
                st.session_state.page_offset += 10
                with st.spinner("Loading more matches..."):
                    new_matches, _ = find_matching_images(
                        df,
                        selected_attributes,
                        batch_size=10,
                        offset=st.session_state.page_offset
                    )
                    st.session_state.current_matches.extend(new_matches)
                st.rerun()
        
        # Show enhancement options if an image is selected
        if st.session_state.selected_image:
            st.markdown("---")
            st.markdown(f"### 🛠️ Enhancing: {os.path.basename(st.session_state.selected_image)}")
            enhance_image(st.session_state.selected_image)
        
        # Show all attributes for debugging
        with st.expander("📊 View All Matching Attributes"):
            st.dataframe(pd.DataFrame([img['attributes'] for img in st.session_state.current_matches]))
    
    elif 'total_matches' in st.session_state and st.session_state.total_matches == 0:
        st.warning("**⚠️ No matches found**")
        st.markdown("Try adjusting your search parameters")

if __name__ == "__main__":
    main()