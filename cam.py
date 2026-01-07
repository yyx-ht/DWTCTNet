import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import numpy as np
from PIL import Image
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ht.model.ours.wtdmcatunet import wtdmcatunet
from ht.model.ours.wtadd import wtadd
from ht.model.ours.wtcontact import wtconcat
from ht.model.dnanet import DNANet
from ht.model.wtnet.ID_UNet import ID_UNet
from ht.model.uiunet import UIUNET
from ht.model.u2net import U2NETP

# Define paths
input_folder = r'D:\PyCharm\projects\DL\ht\test\images'  # Folder containing input images
output_folder = r'D:\PyCharm\projects\DL\camout\wtdmcatunet\fuse3'  # Folder to save CAM images
os.makedirs(output_folder, exist_ok=True)

# Load the model
model = wtdmcatunet(3,2)
save_pth = r'D:\PyCharm\projects\DL\ht\predict1\wtdmcatunet\best.pt'
model.load_state_dict(torch.load(save_pth))
model = model.cuda().eval()

# Define classes
sem_classes = ['background', 'crack']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
car_category = sem_class_to_idx["crack"]

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

# Iterate through all images in the folder
for image_name in os.listdir(input_folder):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue  # Skip non-image files

    # Load and preprocess the image
    image_path = os.path.join(input_folder, image_name)
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image, np.float32) / 255.0
    imag = np.array(image, np.float32).transpose(2, 0, 1)
    input_tensor = torch.Tensor(imag)
    input_tensor = input_tensor.unsqueeze(0).cuda()

    # Forward pass
    x = model(input_tensor)
    probs = torch.softmax(x, dim=1).cpu()

    # Generate the round mask
    round_mask = torch.argmax(probs[0], dim=0).detach().cpu().numpy()
    round_mask_uint8 = 255 * np.uint8(round_mask == car_category)
    round_mask_float = np.float32(round_mask == car_category)

    # Define target layers
    target_layers = [model.lfuse3]
    targets = [SemanticSegmentationTarget(car_category, round_mask_float)]

    # Create CAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True, image_weight=0)

    # Save the CAM image
    output_filename = f"{os.path.splitext(image_name)[0]}.png"
    full_output_path = os.path.join(output_folder, output_filename)

    # Save the image
    img = Image.fromarray(cam_image)
    img.save(full_output_path)
    print(f"Saved CAM image to: {full_output_path}")