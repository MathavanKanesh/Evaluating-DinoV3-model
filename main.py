from pathlib import Path
from PIL import Image
import numpy as np

dataset_path = Path(r"C:\Users\matha\Downloads\Projects\DinoV3_trial\dataset")


# Images path

images_path = dataset_path / "images"

print("First 20 image files:")
print(list(images_path.iterdir())[:20])

image_files = list(images_path.glob("*.png"))

for image in image_files[0:20]:
    print(image)

image_path = image_files[0]

image = Image.open(image_path)

print(image_path, image.size, image.mode) # Images are mode RGB, 1920 x 1080 pixels 
image_array = np.array(image)
print(np.unique(image_array)) # 255 labels 


# Masks path

masks_path = dataset_path / "masks"

print("First 20 mask files:")
print(list(masks_path.iterdir())[:20])

masks_files = list(masks_path.glob("*.png"))

for mask in masks_files[0:20]:
    print(mask)

mask_path = masks_files[0]

mask = Image.open(mask_path)

print(mask_path, mask.size, mask.mode) # Masks  are mode L, 1920 x 1080 pixels

mask_array = np.array(mask)
print(np.unique(mask_array)) # 5 labels

import matplotlib
import matplotlib.pyplot as plt 

plt.imshow(mask)
plt.axis("off")
plt.show()

unique_values, counts = np.unique(mask_array, return_counts = True)

for value, count in zip(unique_values, counts):
    print("Label", value, ":", count, "pixels")
# Label 0: 1393747 pixels, Label 1: 14221 pixels, Label 2: 49435 pixels, Label 3: 613415 pixels, Label 5: 2782 pixels

# Turn mask into array of zeroes 

mask_annotation = np.zeros_like(mask_array) # Creates a numpy array filled with zeroes, same shape as annotation_array
mask_annotation[mask_array == 1] = 0 # 0 is black 
mask_annotation[mask_array == 2] = 255 # 255 is white 
mask_annotation[mask_array == 3] = 128

# Create binary ground truth mask 
ground_truth = np.zeros_like(mask_array)
# Everything that isnt background becomes surgical instrument 
ground_truth[mask_array !=0] = 1
print("Ground truth labels:", np.unique(ground_truth))



#Surgical Image vs Binary ground truth 
plt.figure()

plt.subplot(1, 2, 1) # 1 row, 2 columns, use position 1 
plt.imshow(image)
plt.title("Surgical Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(ground_truth, cmap="gray")
plt.title("Binary Ground Truth")
plt.axis("off")

plt.show()

# Preprocessing; Change image size to 682x384 while maintaining ratios
new_height = 384 
new_width = 683
image_resized = image.resize((new_width, new_height))

# Resize the ground truth 
ground_truth_image = Image.fromarray(ground_truth) # Convert from Numpy array back to an image 
ground_truth_resized = ground_truth_image.resize(
    (new_width, new_height),
    Image.Resampling.NEAREST # NEAREST means dont create any new values 
)

#Visualize resized versions: 
# Visualize resized image and ground truth

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image_resized)
plt.title("Resized Surgical Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(ground_truth_resized, cmap="gray")
plt.title("Resized Binary Ground Truth")
plt.axis("off")

plt.show()


#Prepare image for DinoV3 

# Convert resized image to NumPy Array 
image_resized_array = np.array(image_resized)
print("Image shape:", image_resized_array.shape) #384, 683, 3 --> 384 x 683 pixels with RGB channel 
print("Image data type:", image_resized_array.dtype) # uint8 --> unsigned integer, 8 bits to store each number 

# Convert to a PyTorch tensor: 
import torch 

image_tensor = torch.tensor(image_resized_array) # Convert array into a tensor 
#Tensor Characteristics: 
print("Tensor Shape:", image_tensor.shape) # [384, 683,3]
print("Tensor data type:", image_tensor.dtype) # torch.uint8

#Tensor is still uint8 and dimensions are height, width, channels --> Need to change these to float and channels, height, width

image_tensor = image_tensor.float()
print("Tensor data type now:", image_tensor.dtype) #torch.flot32 now 

image_tensor = image_tensor.permute(2,0,1) # Put channels first, height second, width third
print("Tensor shape now:", image_tensor.shape) # [3,384,683]

# DinoV3's pretrained weights expect image to be normalized (range between 0 and 1), because neural networks generally work beteer when inputs are on smaller, conssitent scale
image_tensor /= 255.0 
print("minimum:", image_tensor.min()) # 0 
print("maximum:", image_tensor.max()) # 1

# DinoV3 pretrained weights

checkpoint_path = Path(
    r"C:\Users\matha\Downloads\Projects\DinoV3_trial\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
REPO_DIR = r"C:\Users\matha\Downloads\Projects\DinoV3_trial\dinov3"

dinov3_vits16 = torch.hub.load(  # Load a pretrained model using PyTorch's model loading system 
    REPO_DIR,          # Tells where the DinoV3 code is located 
    "dinov3_vits16",     #Model architecture 
    source = "local",      #Repository already on my computer 
    weights = str(checkpoint_path)  # Use this checkpoint as pretrained weights 
)

# ViT-S/16 checkpoint using 16x16 pixel patches

image_tensor = image_tensor.unsqueeze(0) #tensor current in form (channels, height, width) : We need batch
print("Tensor shape with batch:", image_tensor.shape)

# Get features 
with torch.no_grad():
    features = dinov3_vits16(image_tensor) # Dont calculate gradient

print("Feature shape:", features.shape)
#  Tensor shape with batch: torch.size [1,3,384,683] # 1 image, 3 colour channels (RGB), 384x683 pixels
# Feature shape: torch.size[1,384] # 1 384 dimensional feature vector 

# Get DinoV3 spatial/patch features 
# Dinov3's ViT-S/16 divides image into 16x16 pixel patches, each patch processed by the transformer and each patch get its own feature vector, thats what gives us spatial info


dinov3_vits16.eval() # Evaluating the model

with torch.no_grad(): 
    patch_features = dinov3_vits16.get_intermediate_layers(   # get_intermediate_Layers method gets features 
        image_tensor,   # Pass the image 
        n=1,               # 1 intermediate layer 
        reshape = True    # Reshape patch representation into a spatial grid 
    )
features = patch_features[0]
print("Patch feature shape:", features.shape) # print dimensions of first feature tensor 
# Patch feature shape: torch.Size([1, 384, 24, 42])  --> DinoV3 transformed 384x683 pixel image into a 24x42 spatial grid, where every location contains 384 learned features (useful numerical representations)


# Simple Segmentation Head: Turn the 384 features (numbers) into one number describing likeliness that the location is a surgical instrument 
import torch.nn as nn  # Import for neural network training 
 
segmentation_head = nn.Conv2d( # nn.Conv2d : Convolutional layer 
    in_channels = 384, 
    out_channels = 1, 
    kernel_size = 1 # 1x1 segmentation head (looks at each spatial location at a time, not 9 like a 3x3 would look at)
)

logit_prediction = segmentation_head(features)
print("Prediction:", logit_prediction.shape)
# [1,1,24,42] now --> Number of "features"/ predictions at each location changed from 3 to 1 


# The number inside the prediction are called "logits", convert logits into probabilities using a "sigmoid". Cant use logits because probabilities must be between 0 and 1
# Sigmoid function: 1/1+e^-x converts logits into numbers between 0 and 1 

prediction = torch.sigmoid(logit_prediction)
print("Prediction shape:", prediction.shape )
print("Prediction min:", prediction.min())
print("Prediction max:", prediction.max())

#Prediction min: tensor(0.4144, grad_fn=<MinBackward1>)
#Prediction max: tensor(0.6104, grad_fn=<MaxBackward1>)
# Both values around 0.5 because the model is untrained 

# Compare prediction to ground truth mask 
#  prediction is only 24x42 while ground truth mask is 384x683, need to upscale prediction 

import torch.nn.functional as F # Pythons function for resizing tensors 
new_width = 683
new_height = 384
prediction_upsampled = F.interpolate(
    prediction, 
    size = (new_height, new_width),
    mode = "bilinear", 
    align_corners = False 

)
print("Upsampled prediction:",  prediction_upsampled)
print("upsampled prediction shape and datatype:", prediction_upsampled.shape) #[1,1,384,683] 

# Make ground truth tensor compatible with prediction tensor (datatype: uint8)


# Convert resized ground truth from PIL image to NumPy array
ground_truth_array = np.array(ground_truth_resized)

# Convert NumPy array to PyTorch tensor
ground_truth_tensor = torch.tensor(ground_truth_array)

print("Ground truth shape:", ground_truth_tensor.shape)
print("Ground truth data type:", ground_truth_tensor.dtype) # uint8 , need to convert to float 

ground_truth_tensor = ground_truth_tensor.unsqueeze(0) #Add batch 
ground_truth_tensor = ground_truth_tensor.unsqueeze(0) # Add channel 

print("Ground truth shape:", ground_truth_tensor.shape)
ground_truth_tensor = ground_truth_tensor.float()

# Right now we have the model's prediction for each pixel, the correct answer for each pixel and both have the same shape 


# Upsample logit_prediction

logit_prediction_upsampled = F.interpolate(
    logit_prediction,
    size=(new_height, new_width),
    mode="bilinear",
    align_corners=False
)

print("Upsampled logits shape:", logit_prediction_upsampled.shape)

#BCEwithLogitsLoss: 
criterion = nn.BCEWithLogitsLoss()

loss = criterion (
    logit_prediction_upsampled,   # Takes in ground truth tensor and logits as input to calculate loss
    ground_truth_tensor
)
print("Loss:", loss.item()) #Loss: 0.724... (Not good)


# Backpropogation: Currently random weights, change the weights so loss becomes smaller 

#loss.backward() method : Pytorch calculates gradients. gradients tell us: "Increase/Decrease the weight this much increases/decreases loss this much"
loss.backward()
print("Weight gradient:", segmentation_head.weight.grad) # Give current weights of segmentation head  
print("Bias gradient:", segmentation_head.bias.grad) # Give gradient that pyTorch calculated for these weights --> 0.1654, decreasing bias should reduce loss 

#Optimizer: CHanges the weights 

optimizer = torch.optim.SGD( # PyTorch optimization algorithms , specific one we are using is Stochastic Gradient Descent 
    segmentation_head.parameters(), # Give optimizer all learnable parameters in segmentation head
    lr = 0.01 # Learning rate , how large of a step optimizer takes when changing parameters 
)
optimizer.step()
# After an update gradients are still stored, before next training iteration clear them using: optimizer.zero_grad()


# Run the forward pass again and calculate the loss using new weights 
with torch.no_grad():
    patch_features = dinov3_vits16.get_intermediate_layers(
        image_tensor,
        n=1,
        reshape=True
    )

features = patch_features[0]

logit_prediction = segmentation_head(features)

logit_prediction_upsampled = F.interpolate(
    logit_prediction,
    size=(new_height, new_width),
    mode="bilinear",
    align_corners=False
)

loss_after = criterion(
    logit_prediction_upsampled,
    ground_truth_tensor
)

print("Loss after update:", loss_after.item()) # 0.67428... now instead of 0.724...

# Clear old gradients 
optimizer.zero_grad()

# Train on 20 images now 

