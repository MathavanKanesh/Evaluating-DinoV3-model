#Previously: DinoV3 head was frozen while segmentation head learned 
# Now: Dinov3 head and segmentation head learn #Instead of training on all 20 images split the dataset --> Train on 16 images and validate on 4 images , see how well the model generalizes 
#Remove torch.nograd() from training loop so that gradients are calculated through DINOv3, so its weights can be updated during training
# Give the optimizer DinoV3 parameters as well now, not just segmentation head parameters 

#Imports: 
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Get images and masks
dataset_path = Path(
    r"C:\Users\matha\Downloads\Projects\DinoV3_trial\dataset"
)

images_path = dataset_path / "images"
masks_path = dataset_path / "masks"

image_files = list(images_path.glob("*.png"))
masks_files = list(masks_path.glob("*.png"))

#Split into training and evaluating set 
training_image_files = image_files[:16]
training_mask_files = masks_files[:16]

validation_image_files = image_files[16:]
validation_masks_files = masks_files[16:]

print("Number of images:", len(image_files))
print("Number of masks:", len(masks_files))

#Add DinoV3
checkpoint_path = Path(
    r"C:\Users\matha\Downloads\Projects\DinoV3_trial\dinov3_vits16_pretrain_lvd1689m-08c60483.pth" # Where DinoV3 pretrained weights are stored
)
REPO_DIR = r"C:\Users\matha\Downloads\Projects\DinoV3_trial\dinov3" # Where DinoV3 code is located 

dinov3_vits16 = torch.hub.load(  # Load a pretrained model using PyTorch's model loading system 
    REPO_DIR,          # Tells where the DinoV3 code is located 
    "dinov3_vits16",     #Model architecture 
    source = "local",      #Repository already on my computer 
    weights = str(checkpoint_path)  # Use this checkpoint as pretrained weights 
)
dinov3_vits16.eval()

# Add Segmentation Head
segmentation_head = nn.Conv2d( # nn.Conv2d : Convolutional layer 
    in_channels = 384, 
    out_channels = 1, 
    kernel_size = 1 # 1x1 segmentation head (looks at each spatial location at a time, not 9 like a 3x3 would look at)
)


criterion = nn.BCEWithLogitsLoss() # Calculates how wrong prediction is 

optimizer = torch.optim.SGD(  # Updates segmentation head and DinoV3 weights now
    list(segmentation_head.parameters()) + list(dinov3_vits16.parameters()),
    lr=0.01
)
# Start training loop 

image_width = 683 
image_height = 384

for epoch in range(5): # an epoch is one complete pass through the entire training datase

    total_loss = 0

    for i in range(len(training_image_files)):

        # Load image
        image = Image.open(training_image_files[i]).convert("RGB") # Convert all image files to RGB
        image = image.resize((683, 384)) # Resize image 

        image_array = np.array(image) # Numpy Array 
        image_tensor = torch.tensor(image_array) # PyTorch Tensor 
        image_tensor = image_tensor.float() # Convert from uint8 to float 
        image_tensor = image_tensor / 255.0 # Scales pixel's values from 0-255 to 0-1 (normalizing the pixel range)
        image_tensor = image_tensor.permute(2, 0, 1) # Convert to format C  x H xW 
        image_tensor = image_tensor.unsqueeze(0) # Add Batch, now : B x C x H x W 

        # Load mask
        mask = Image.open(training_mask_files[i]).convert("L") # Convert mask files to "Luminance" (Grayscale)
        mask = mask.resize((683, 384), Image.Resampling.NEAREST) # Resize mask , "Image.Resampling.NEAREST" preserves mask values

        mask_array = np.array(mask) # Numpy Array 
        ground_truth_array = (mask_array > 0).astype(np.float32) # Turn into binary mask 

        ground_truth_tensor = torch.tensor(ground_truth_array)
        ground_truth_tensor = ground_truth_tensor.unsqueeze(0)
        ground_truth_tensor = ground_truth_tensor.unsqueeze(0)

        # Get DINOv3 features
        #with torch.no_grad(): --> Remove this 
        patch_features = dinov3_vits16.get_intermediate_layers(
                image_tensor,
                n=1,
                reshape=True
            )

        features = patch_features[0]

        # Prediction
        logit_prediction = segmentation_head(features)

        # Resize prediction to ground truth size
        logit_prediction_upsampled = F.interpolate(
            logit_prediction,
            size=(384, 683),
            mode="bilinear",
            align_corners=False
        )

        # Calculate loss
        loss = criterion(
            logit_prediction_upsampled,
            ground_truth_tensor
        )

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

    average_loss = total_loss / len(training_image_files) # Average loss across all images 

    print("Epoch:", epoch + 1, "Average loss:", average_loss)

#Epoch: 1 Average loss: 0.44203594885766506
#Epoch: 2 Average loss: 0.27274351473897696
#Epoch: 3 Average loss: 0.19728925358504057
#Epoch: 4 Average loss: 0.1519598918966949
#Epoch: 5 Average loss: 0.12469492806121707



# Convert logits into probabilities 
prediction = torch.sigmoid(logit_prediction_upsampled)
# Turn probabilities into values between 0 and 1 

prediction_binary = (prediction > 0.5).float()


#Validation Loop: Dont update weights (no loss.backward, optimizer.step because evaluating now, not training anymore)

total_dice_score = 0 

for i in range(len(validation_image_files)):

    # Load image

    image = Image.open(validation_image_files[i]).convert("RGB")

    image = image.resize((683, 384))

    image_array = np.array(image)

    image_tensor = torch.tensor(image_array)

    image_tensor = image_tensor.float()

    image_tensor = image_tensor / 255.0

    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor = image_tensor.unsqueeze(0)

    # Load mask

    mask = Image.open(validation_masks_files[i]).convert("L")  

    mask = mask.resize((683, 384), Image.Resampling.NEAREST)

    mask_array = np.array(mask)

    ground_truth_array = (mask_array > 0).astype(np.float32)

    ground_truth_tensor = torch.tensor(ground_truth_array)

    ground_truth_tensor = ground_truth_tensor.unsqueeze(0)

    ground_truth_tensor = ground_truth_tensor.unsqueeze(0)

    # Get DINOv3 features

    with torch.no_grad():

        patch_features = dinov3_vits16.get_intermediate_layers(
            image_tensor,
            n=1,
            reshape=True
        )

        features = patch_features[0]

        # Get prediction

        logit_prediction = segmentation_head(features)

        logit_prediction_upsampled = F.interpolate(
            logit_prediction,
            size=(384, 683),
            mode="bilinear",
            align_corners=False
        )

        prediction = torch.sigmoid(logit_prediction_upsampled)

    # Convert prediction to binary mask

    prediction_binary = (prediction > 0.5).float()

    #Calculate the overlap: 

    intersection = (prediction_binary * ground_truth_tensor).sum() # Compares the two masks pixel by pixel 

    # Count predicted and ground truth pixels: 

    prediction_sum = prediction_binary.sum()

    ground_truth_sum = ground_truth_tensor.sum()

    #Dice Score Calculation: Recall: Dice Score = 2|Prediction ∩ GroundTruth| / |Prediction| + |Ground Truth| 

    # The result is between 0 and 1 

    dice = (2 * intersection) / (prediction_sum + ground_truth_sum)

    total_dice_score = total_dice_score + dice.item() # dice.item() gives actual value stored inside single value tensor


# Calculate the average Dice Score across all images

average_dice_score = total_dice_score / len(validation_image_files)

print("Average Dice Score:", average_dice_score)

# Average Dice Score: 0.80631... after finetuning

#Visualize one image and its mask 
import matplotlib.pyplot as plt
plt.figure()

plt.imshow(image_array)
plt.axis("off")
plt.title("Image")

plt.figure()

plt.imshow(mask_array)
plt.axis("off")
plt.title("Mask")

plt.show()

