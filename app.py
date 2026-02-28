import gradio as gr
import numpy as np
from PIL import Image
from predict import load_model, predict_image, explain_with_lime
import tempfile, os

model = load_model()

def analyse(image):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        Image.fromarray(image).save(tmp.name)
        result = predict_image(model, tmp.name)
        os.unlink(tmp.name)
    label = result['prediction']
    conf  = result['confidence']
    color = '#FF4B4B' if label == 'Fake' else '#21B36B'
    summary = f'**{label}** ({conf*100:.1f}% confidence)'
    breakdown = f"Real: {result['real_prob']*100:.1f}%  |  Fake: {result['fake_prob']*100:.1f}%"
    return summary, breakdown

gr.Interface(
    fn=analyse,
    inputs=gr.Image(label='Upload face image'),
    outputs=[
        gr.Markdown(label='Detection Result'),
        gr.Textbox(label='Probability Breakdown')
    ],
    title='Deepfake Detection System',
    description='EfficientNet-B0 fine-tuned for deepfake detection with LIME explainability',
    examples=['examples/real_face.jpg', 'examples/deepfake_face.jpg']
).launch()
