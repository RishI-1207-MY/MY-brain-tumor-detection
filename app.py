import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Brain Tumor AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ==================================
# CUSTOM CSS
# ==================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fc;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    color: #0f172a;
}

.subtitle {
    text-align: center;
    color: #475569;
    font-size: 20px;
}

.metric-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
    text-align: center;
}

.prediction-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 3px 15px rgba(0,0,0,0.1);
}

.sidebar-header {
    color: white;
    font-size: 24px;
    font-weight: bold;
}

section[data-testid="stSidebar"] {
    background: #1e40af;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# LOAD MODEL
# ==================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_brain_tumor_model.keras"
    )

model = load_model()

# ==================================
# CLASS NAMES
# ==================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ==================================
# SIDEBAR
# ==================================

st.sidebar.markdown(
    "<div class='sidebar-header'>🧠 Brain Tumor AI</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Model Information

**Architecture**
- EfficientNetB0

**Framework**
- TensorFlow

**Classes**
- Glioma
- Meningioma
- No Tumor
- Pituitary
""")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Developer

Rishi Khandelwal

AI & Deep Learning Project
""")

# ==================================
# HEADER
# ==================================

st.markdown(
    "<div class='main-title'>🧠 Brain Tumor AI Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>MRI Classification using Deep Learning</div>",
    unsafe_allow_html=True
)

st.write("")
st.write("")

# ==================================
# TOP METRICS
# ==================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-box">
    <h2>4</h2>
    <p>Tumor Classes</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-box">
    <h2>224×224</h2>
    <p>Input Size</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-box">
    <h2>AI</h2>
    <p>EfficientNetB0</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ==================================
# MAIN CONTENT
# ==================================

left, right = st.columns([1,1])

with left:

    st.subheader("📤 Upload MRI Image")

    uploaded_file = st.file_uploader(
        "Select MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded MRI Scan",
            use_container_width=True
        )

# ==================================
# PREDICTION
# ==================================

if uploaded_file:

    image = Image.open(uploaded_file)

    img = image.resize((224,224))

    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack(
            (img_array,) * 3,
            axis=-1
        )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction)
    )

    with right:

        st.subheader("🔍 AI Diagnosis")

        st.markdown(
            f"""
            <div class='prediction-card'>
            <h2>{predicted_class}</h2>
            <h3>Confidence: {confidence*100:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(confidence)

        st.write("")

        st.subheader("📊 Probability Distribution")

        df = pd.DataFrame({
            "Class": class_names,
            "Probability": prediction[0]
        })

        fig = px.bar(
            df,
            x="Class",
            y="Probability",
            text_auto=".2%"
        )

        fig.update_layout(
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("🩺 Recommendation")

        if predicted_class == "No Tumor":
            st.success(
                "No tumor detected."
            )

        elif predicted_class == "Glioma":
            st.error(
                "Possible Glioma detected. Please consult a neurologist."
            )

        elif predicted_class == "Meningioma":
            st.warning(
                "Possible Meningioma detected. Further medical evaluation recommended."
            )

        elif predicted_class == "Pituitary":
            st.warning(
                "Possible Pituitary Tumor detected. Specialist consultation advised."
            )

# ==================================
# FOOTER
# ==================================

st.markdown("---")

st.markdown(
    """
    <center>
    <b>Developed by Rishi Khandelwal</b><br>
    Brain Tumor Detection using Deep Learning & EfficientNetB0
    </center>
    """,
    unsafe_allow_html=True
)