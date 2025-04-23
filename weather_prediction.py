import torch
import torchvision.transforms as transforms
from PIL import Image
from config import Common

def predict(frame, modelPath):
    '''
    Predict the weather condition based on a frame from the video.
    :param frame: Frame from video
    :param modelPath: Path to the trained weather model
    :return: Predicted weather label
    '''
    # Convert frame (NumPy array) to PIL image
    image = Image.fromarray(frame)
    # Resize image to match model input size
    image = image.resize(Common.imageSize)

    model = torch.load(modelPath, map_location=Common.device)
    model = model.to(Common.device)

    # Transform image to tensor
    transform = transforms.ToTensor()
    x = transform(image)
    x = torch.unsqueeze(x, 0)
    x = x.to(Common.device)

    output = model(x)

    # Use argmax to get the class with the highest probability
    output = torch.argmax(output)
    predicted_label = Common.labels[output.item()]

    return predicted_label

