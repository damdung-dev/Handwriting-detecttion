from django.shortcuts import render
from django.http import JsonResponse
from .forms import UploadImageForm
from PIL import Image, ImageOps
import io, base64, torch
from mnist_model import CNNModel  # import class CNNModel
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (dùng model epoch cuối cùng)
model = CNNModel().to(device)
model.load_state_dict(torch.load('saved_model/mnist_cnn_epoch20.pth', map_location=device))
model.eval()

# Transform dùng cho canvas/upload
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ==========================
# Hiển thị index.html
# ==========================
def index(request):
    form = UploadImageForm()
    return render(request, 'detection_app/index.html', {'form': form})

# ==========================
# Canvas
# ==========================
def predict_draw(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        img_data = data['image'].split(',')[1]
        img_bytes = io.BytesIO(base64.b64decode(img_data))
        img = Image.open(img_bytes).convert('L')
        img = ImageOps.invert(img)  # đảo màu
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True).item()
        return JsonResponse({'pred': pred})
    return JsonResponse({'error':'POST request required'})

# ==========================
# Upload file
# ==========================
def predict_file(request):
    result = None
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            img = Image.open(form.cleaned_data['image']).convert('L')
            img = ImageOps.invert(img)
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img)
                probs = torch.exp(output)  # vì model trả log_softmax
                pred = output.argmax(dim=1).item()
                confidence = probs[0, pred].item()  # xác suất của lớp dự đoán

    return render(request, 'detection_app/results.html', {
    'form': UploadImageForm(),
    'result': pred,
    'confidence': round(confidence*100, 2)  # hiển thị % xác suất
})

