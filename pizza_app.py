import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

# Configurations
CLASS_NAMES = ['basil', 'mushroom', 'pepper', 'pepperoni', 'pineapple', 'sausage']
MODEL_PATH = 'best_model.pth'

st.set_page_config(
  page_title="Pizza Scan", 
  page_icon="🍕", 
  layout="wide",
  initial_sidebar_state="collapsed"
)

# Helpers
def img_to_base64(img):
  buffered = io.BytesIO()
  if img.mode == 'RGBA':
    img = img.convert('RGB')
  img.save(buffered, format="JPEG")
  return base64.b64encode(buffered.getvalue()).decode()

def reset_app():
  st.session_state.result = None
  st.session_state.show_upload = False
  st.session_state.camera_id += 1 

# Model loading
@st.cache_resource
def load_model():
  model = models.resnet18(weights=None)
  model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
  model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
  model.eval()
  return model

model = load_model()

# Preprocessing
inference_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
  if image.mode != "RGB":
    image = image.convert("RGB")
  
  img_t = inference_transform(image)
  batch_t = torch.unsqueeze(img_t, 0)
  
  with torch.no_grad():
    out = model(batch_t)
    probabilities = torch.nn.functional.softmax(out, dim=1)[0]
    prob, index = torch.max(probabilities, 0)
    
  return CLASS_NAMES[index], prob.item()

# Session state initialization
if 'result' not in st.session_state:
  st.session_state.result = None
if 'show_upload' not in st.session_state:
  st.session_state.show_upload = False
if 'camera_id' not in st.session_state:
  st.session_state.camera_id = 0
if 'show_info' not in st.session_state:
  st.session_state.show_info = True

# CSS styling
st.markdown("""
<style>
  .stApp {
    background-color: black !important;
    user-select: none !important;
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
  }
  header, footer, [data-testid="stHeader"] { display: none !important; }
  .block-container {
    padding: 0 !important; margin: 0 !important;
    max-width: 100% !important; background: transparent !important;
  }
  a.anchor-link {
    display: none !important;
  }
  [data-testid="stMarkdownContainer"] a {
    display: none !important;
  }
  video {
    position: fixed !important; top: 0; left: 0;
    width: 100vw !important; height: 100vh !important;
    object-fit: cover !important; z-index: 1 !important;
  }
  [data-testid="stCameraInput"] button {
    position: fixed !important; bottom: 40px !important; left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 10 !important;
    width: 80px !important; height: 80px !important;
    border-radius: 50% !important; border: 4px solid white !important;
    background: transparent !important; color: transparent !important;
  }
  [data-testid="stCameraInput"] button::after {
    content: ''; display: block; width: 65px; height: 65px;
    background: rgba(255, 255, 255, 0.9); border-radius: 50%;
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  }
  button[kind="tertiary"] {
    position: fixed !important; top: 20px !important; right: 20px !important;
    z-index: 20 !important;
    width: 50px !important; height: 50px !important;
    border-radius: 50% !important; border: 2px solid white !important;
    background: rgba(0, 0, 0, 0.5) !important; color: white !important;
    font-size: 24px !important;
  }
  .info-popup {
    position: fixed !important; top: 50% !important; left: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: 90% !important; max-width: 400px !important;
    height: auto !important; min-height: 300px !important;
    background: rgba(20, 20, 20, 0.95) !important;
    padding: 40px !important; border-radius: 20px !important;
    border: 1px solid #444 !important;
    z-index: 50 !important;
    color: white !important; text-align: center !important;
    display: flex !important; flex-direction: column !important; justify-content: center !important;
  }
  [data-testid="stFileUploader"] {
    position: fixed !important; top: 50% !important; left: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: 90% !important; 
    height: auto !important;
    max-width: 700px !important;
    background: rgba(20, 20, 20, 0.95) !important;
    padding: 40px !important; border-radius: 20px !important;
    border: 1px solid #444 !important;
    z-index: 50 !important;
    text-align: center !important;
  }
  [data-testid="stFileUploader"] label { color: white !important; font-size: 1.1rem; }
  div.stButton > button[kind="primary"] {
    position: fixed !important;
    bottom: 50px !important; left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 999999 !important;
    width: auto !important; height: auto !important;
    padding: 15px 40px !important;
    border-radius: 30px !important;
    font-size: 18px !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.5) !important;
    background-color: white !important; color: black !important; border: none !important;
  }
</style>
""", unsafe_allow_html=True)

# App logic
camera_key = f"camera_{st.session_state.camera_id}"
camera_buffer = st.camera_input("Camera", label_visibility="hidden", key=camera_key)

if camera_buffer and not st.session_state.result and not st.session_state.show_info:
  final_image = Image.open(camera_buffer)
  label, conf = predict_image(final_image)
  st.session_state.result = (label, conf, final_image)
  st.rerun()

# Info display
if st.session_state.show_info:
  st.markdown("""
  <style>
    div.stButton > button[kind="secondary"] {
      position: fixed !important;
      top: 50% !important; left: 50% !important;
      transform: translate(150px, -190px) !important;
      z-index: 60 !important;
      width: 40px !important; height: 40px !important;
      border-radius: 50% !important;
      background: rgba(255, 50, 50, 0.8) !important; 
      color: white !important; border: none !important;
      font-size: 16px !important; padding: 0 !important;
    }
    [data-testid="stCameraInput"] button {
      display: none !important;
      opacity: 0 !important;
      pointer-events: none !important;
    }
  </style>
  """, unsafe_allow_html=True)
  
  with st.container():
    st.markdown("""
    <div class="info-popup">
      <h2 style="margin-bottom: 20px;">🍕 Welcome to Pizza Scan 🍕</h2>
      <p style="line-height: 1.6; color: #ccc; font-size: 18px;">
        <b>1.</b> Take a picture of a pizza.<br>
        <b>2.</b> Or upload an image.<br>
        <b>3.</b> The app will identify the topping!<br>
        <br>
        <i>Toppings: Pepperoni, Sausage, Pineapple, Peppers, Mushrooms, Basil</i>
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✕", key="close_info_btn", type="secondary"):
      st.session_state.show_info = False
      st.rerun()

# Result display
elif st.session_state.result:
  label, conf, img = st.session_state.result
  img_str = img_to_base64(img)
  
  html_code = f"""
  <div style="
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background-color: black; z-index: 99990;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    pointer-events: none;
  ">
    <img src="data:image/jpeg;base64,{img_str}" style="
      max-width: 90%; max-height: 60vh; 
      border-radius: 15px; border: 2px solid #444;
      object-fit: contain; pointer-events: auto;
    " />
    <h2 style="color: white; margin-top: 20px; font-family: sans-serif; pointer-events: auto;">
      {label.upper()} ({(conf*100):.2f}%)
    </h2>
  </div>
  """
  st.markdown(html_code, unsafe_allow_html=True)
  st.button("Scan Again", type="primary", on_click=reset_app)

# Upload mode
elif st.session_state.show_upload:
  st.markdown("""
  <style>
    div.stButton > button[kind="secondary"] {
      position: fixed !important;
      top: 50% !important; left: 50% !important;
      transform: translate(300px, -80px) !important;
      z-index: 60 !important;
      width: 40px !important; height: 40px !important;
      border-radius: 50% !important;
      background: rgba(255, 50, 50, 0.8) !important; 
      color: white !important; border: none !important;
      font-size: 16px !important; padding: 0 !important;
    }
    [data-testid="stCameraInput"] button {
      display: none !important;
    }
  </style>
  """, unsafe_allow_html=True)

  if st.button("✕", key="close_upload_btn", type="secondary"):
    st.session_state.show_upload = False
    st.rerun()

  uploaded_file = st.file_uploader("Select an image", type=['jpg', 'jpeg', 'png'])
  
  if uploaded_file:
    final_image = Image.open(uploaded_file)
    label, conf = predict_image(final_image)
    st.session_state.result = (label, conf, final_image)
    st.session_state.show_upload = False
    st.rerun()

# Standard camera mode
else:
  st.markdown("""
  <style>
    div.stButton > button[kind="secondary"] {
      position: fixed !important; bottom: 50px !important; right: 30px !important;
      z-index: 20 !important;
      width: 60px !important; height: 60px !important;
      border-radius: 50% !important; border: 2px solid white !important;
      background: rgba(0, 0, 0, 0.5) !important; color: white !important;
      font-size: 24px !important; border-color: white !important;
      transform: none !important;
    }
  </style>
  """, unsafe_allow_html=True)

  if st.button("📁", key="open_upload", type="secondary"):
    st.session_state.show_upload = True
    st.rerun()
    
  if st.button("ℹ️", key="open_info", type="tertiary"):
    st.session_state.show_info = True
    st.rerun()