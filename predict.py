import torch, numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0
from lime import lime_image
from PIL import Image
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LABELS = ['Real', 'Fake']

def load_model(path='deepfake_detector.pth'):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval(); return model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def predict_image(model, img_path: str) -> dict:
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return {'prediction': LABELS[probs.argmax()],
            'confidence': float(probs.max()),
            'real_prob': float(probs[0]),
            'fake_prob': float(probs[1])}

def explain_with_lime(model, img_path: str):
    explainer = lime_image.LimeImageExplainer()
    img = np.array(Image.open(img_path).convert('RGB').resize((224, 224)))
    def batch_predict(images):
        tensors = torch.stack([transform(Image.fromarray(i)) for i in images])
        with torch.no_grad():
            return torch.softmax(model(tensors.to(DEVICE)), dim=1).cpu().numpy()
    explanation = explainer.explain_instance(
        img, batch_predict, top_labels=2, num_samples=500)
    return explanation
