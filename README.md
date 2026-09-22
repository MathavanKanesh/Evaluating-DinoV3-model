Practiced evaluating and fine tuned a pretrained DINOv3 model for surgical image segmentation.

Overview of Dinov3 model: 
Dinov3: Self supervised vision foundation model designed to produce extremely strong visual features, especially dense, spatially meaningful features, without needing task-specific labels during pretraining 

Traditional computer vision models are trained for one task, and need large amount of task-specific labeled data 

Ex) Medical segmentation requires many annotated frames and medical annotation is expensive 

Key idea of Dinov3: Learn a general purpose representation first, then reuse it for many downstream tasks 

Dinov3 doesn’t rely on one loss, major components are: 
Dino loss: Self distillation between teacher and student
iBOT loss: Masked-image modelling 
KoLeo regularization: Encourages useful diversity in the representations
Gram anchoring : Major Dinov3 innovation for preserving dense feature quality

--> Combines multiple types of losses to capture both global and local details 


Main.py: Training on all 20 images

Train.py: Split into training set and validation set

Train_finetune.py: Allowed both DinoV3 and segmentation head to update during training 
