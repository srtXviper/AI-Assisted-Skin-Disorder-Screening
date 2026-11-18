from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import SkinCNN

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'models' / 'skin_cnn.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(page_title='AI-Assisted Skin Screening', page_icon='🔬', layout='centered')
st.title('AI-Assisted Skin Disorder Screening')
st.caption('Educational prototype • decision support only • not a medical diagnosis')
st.warning('This prototype must not be used to diagnose, treat, or rule out a skin condition. Consult a qualified healthcare professional.')

if not CHECKPOINT.exists():
    st.error(f'CNN checkpoint is missing: {CHECKPOINT.relative_to(ROOT)}')
    st.markdown('**To enable predictions:**')
    st.code('1. Add labelled images to dataset/train/<class_name>/\n2. Install dependencies: pip install -r requirements.txt\n3. Train: python train_cnn.py\n4. Restart: streamlit run app.py', language='text')
    st.info('No fake or randomly initialised model is loaded. Inference stays disabled until a compatible trained checkpoint exists.')
    st.stop()

@st.cache_resource
def load_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    classes = ckpt['classes']; size = int(ckpt.get('image_size', 128))
    model = SkinCNN(len(classes)); model.load_state_dict(ckpt['state_dict']); model.to(DEVICE); model.eval()
    tfm = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return model, classes, tfm

try:
    model, classes, transform = load_model()
except Exception as exc:
    st.error(f'Could not load the checkpoint. Ensure it was produced by this project\'s train_cnn.py. Details: {exc}')
    st.stop()

uploaded = st.file_uploader('Upload a skin image', type=['jpg','jpeg','png','webp'])
if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded image', use_container_width=True)
    if st.button('Analyse image', type='primary'):
        with torch.no_grad():
            probs = torch.softmax(model(transform(image).unsqueeze(0).to(DEVICE)), dim=1)[0]
        confidence, index = torch.max(probs, dim=0)
        score = float(confidence.item())
        st.subheader('Model output (experimental)')
        st.write(f'Predicted training label: **{classes[int(index)]}**')
        st.write(f'Model confidence: **{score*100:.1f}%**')
        if score >= 0.85: st.info('High model confidence — still requires professional assessment.')
        elif score >= 0.60: st.warning('Moderate confidence — further clinical evaluation is needed.')
        else: st.error('Uncertain output — do not rely on this prediction; seek professional evaluation.')
        st.caption('Confidence is not diagnostic accuracy or probability that the person has the condition. Thresholds are illustrative, not clinically validated.')
