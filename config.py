import torch
# Weather recognition configuration

class Common:
    basePath = ""  # Image file base path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imageSize = (224,224)
    labels = ["cloudy","haze","rainy","shine","snow","sunny","sunrise","thunder"]



