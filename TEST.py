
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib

# Load the trained model
tl_model = torch.load(r'.\CFExp\weights.pt')
# Set the model to evaluate mode
tl_model.eval()

# Read a sample image and mask from the data-set
ino = 2
img = cv2.imread(f'./CrackForest/Images/{ino:03d}.jpg').transpose(2, 0, 1).reshape(1, 3, 320, 480)

# added towards cuda usage
if torch.cuda.is_available():
    tl_model = tl_model.cuda(0)

# Perform inference on the sample image
with torch.no_grad():
    a = tl_model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
b = a['out'].cpu().detach().numpy()[0][0] > 0.2
# Convert boolean output image to binary
c = b*1

# Plot the output mask
# plt.imshow(c, 'gray', vmin=0, vmax=1)
# # plt.show()
# # plt.pause(100)
# # plt.close()
