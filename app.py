import cv2  
import tensorflow as tf 
import streamlit as st

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
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #0f172a
    );
    color:white;
}

/* HERO */

.hero{
    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.25),
        rgba(139,92,246,0.25)
    );

    backdrop-filter: blur(20px);

    border:1px solid rgba(255,255,255,0.15);

    padding:40px;

    border-radius:30px;

    text-align:center;

    margin-bottom:30px;

    box-shadow:
    0px 10px 30px rgba(0,0,0,0.3);
}

.hero h1{
    font-size:60px;
    color:white;
    font-weight:700;
}

.hero p{
    color:#cbd5e1;
    font-size:20px;
}

/* GLASS CARD */

.glass-card{
    background:rgba(255,255,255,0.08);

    backdrop-filter: blur(20px);

    border:1px solid rgba(255,255,255,0.1);

    border-radius:25px;

    padding:25px;

    box-shadow:
    0px 10px 30px rgba(0,0,0,0.25);

    transition:0.3s;
}

.glass-card:hover{
    transform:translateY(-5px);
}

/* METRIC */

.metric-card{

    background:linear-gradient(
    135deg,
    #2563eb,
    #7c3aed
    );

    border-radius:25px;

    padding:25px;

    text-align:center;

    color:white;

    box-shadow:
    0px 10px 25px rgba(0,0,0,0.3);
}

.metric-card h1{
    font-size:40px;
}

.metric-card p{
    font-size:18px;
}

/* Prediction */

.prediction-card{

    background:linear-gradient(
    135deg,
    #059669,
    #10b981
    );

    padding:30px;

    border-radius:25px;

    color:white;

    text-align:center;

    box-shadow:
    0px 10px 30px rgba(0,0,0,0.3);
}

/* Sidebar */

section[data-testid="stSidebar"]{

    background:
    linear-gradient(
    180deg,
    #1e3a8a,
    #0f172a
    );
}

section[data-testid="stSidebar"] *{
    color:white !important;
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
def make_gradcam_heatmap(img_array, model):

    efficientnet = model.get_layer("efficientnetb0")

    feature_extractor = tf.keras.Model(
        efficientnet.input,
        efficientnet.output
    )

    with tf.GradientTape() as tape:

        img_tensor = tf.convert_to_tensor(
            img_array,
            dtype=tf.float32
        )

        features = feature_extractor(
            img_tensor,
            training=False
        )

        tape.watch(features)

        x = model.layers[3](features)
        x = model.layers[4](x, training=False)
        x = model.layers[5](x)
        x = model.layers[6](x, training=False)
        predictions = model.layers[7](x)

        pred_index = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_channel,
        features
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    features = features[0]

    heatmap = tf.reduce_sum(
        features * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()

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

st.markdown("""
<div class="hero">

<h1>🧠 Brain Tumor AI</h1>

<p>
Deep Learning Powered MRI Diagnosis Platform
</p>

</div>
""", unsafe_allow_html=True)
#brain banner image
st.image(
    "https://images.unsplash.com/photo-1532187643603-ba119ca4109e",
    use_container_width=True
)
st.write("")
st.write("")

# ==================================
# TOP METRICS
# ==================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='metric-card'>
    <h1>4</h1>
    <p>Tumor Classes</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='metric-card'>
    <h1>224</h1>
    <p>Input Resolution</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-card'>
    <h1>AI</h1>
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
    heatmap = make_gradcam_heatmap(
        img_array,
        model
    )

    heatmap = cv2.resize(
        heatmap,
        (224,224)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original = img_array[0].astype(
        np.uint8
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction)
    )

    with right:

        st.subheader("🔍 AI Diagnosis")

        st.markdown(f"""
        <div class='prediction-card'>

        <h1>{predicted_class}</h1>

        <h2>
        Confidence:
        {confidence*100:.2f}%
        </h2>

        </div>
        """, unsafe_allow_html=True)

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
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("🔥 Grad-CAM Explainability")

    st.image(
        overlay,
        caption="Areas focused by the AI model",
        use_container_width=True
    )

    st.subheader("🩺 Recommendation")

    if predicted_class == "No Tumor":
        st.success("No tumor detected.")

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
st.markdown("""
<hr>

<center>

<h4>
Developed by Rishi Khandelwal
</h4>

<p>
Deep Learning • TensorFlow • EfficientNetB0 • Explainable AI
</p>

</center>
""", unsafe_allow_html=True)