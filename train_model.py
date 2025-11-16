from model_definitions import MultiInputSnapCalCNN
from dataset_utils import NutritionDataset
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import numpy as np, os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define file paths (assuming these are correct from previous steps)
img_folder = "/content/drive/MyDrive/kevin/nutrition5k_cleaned/images/"
csv_file = "/content/drive/MyDrive/kevin/nutrition5k_cleaned/labels.csv"

# Define the transform for RGB images (ImageNet stats)
transform_rgb = Compose([
    Resize((448, 448)),  # match timm backbones
    ConvertImageDtype(torch.float32),
    Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# Define a simple resize transform for 1-channel images
mono_transform = Compose([
    Resize((448, 448)),
    ConvertImageDtype(torch.float32),
    # No normalization here, handle in model or add separate mono normalization
])

# Instantiate dataset
dataset = NutritionDataset(csv_file, img_folder, transform_rgb=transform_rgb, mono_transform=mono_transform)

# Check if the dataset is empty
print(f"Dataset size after filtering: {len(dataset)}")

if len(dataset) > 0:
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders (reduced num_workers for debugging)
    train_loader = data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0) # Reduced batch size and num_workers
    val_loader = data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0) # Reduced batch size and num_workers

    # Verify output shapes (optional, as done in a previous cell)
    # images_rgb, images_heat, images_depth, labels, dish_ids = next(iter(train_loader))
    # print("Image RGB batch shape:", images_rgb.shape)
    # print("Image Heat batch shape:", images_heat.shape)
    # print("Image Depth batch shape:", images_depth.shape)
    # print("Label batch shape:", labels.shape)
    # print("Dish IDs shape:", dish_ids.shape)

# Assume the following are already defined and available from previous cells:
# criterion (nn.MSELoss)
# checkpoint_dir, best_model_path, best_val_loss, start_epoch (for checkpointing)

# Directory to save checkpoints
checkpoint_dir = "/content/drive/MyDrive/nutrition_model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_loss = float('inf')  # initialize best validation loss
best_model_path = os.path.join(checkpoint_dir, "best_multi_input_model.pt") # Use a distinct name
start_epoch = 0 # initialize start epoch

# Instantiate the multi-input model
model = MultiInputSnapCalCNN()
model.to(device)

# Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# === Optionally Resume from Best Checkpoint ===
# Resume from the best checkpoint if it exists and resume_from_best is True
resume_from_best = True # Set to False if you want to start fresh
if resume_from_best and os.path.exists(best_model_path):
    print(f"Attempting to load checkpoint from {best_model_path}")
    try:
        # Load the checkpoint
        checkpoint = torch.load(best_model_path)

        # Load model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state dictionary (if saved)
        if 'optimizer_state_dict' in checkpoint:
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state dictionary (if saved)
        if 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load best validation loss and starting epoch
        if 'best_val_loss' in checkpoint:
             best_val_loss = checkpoint['best_val_loss']
        if 'epoch' in checkpoint:
             start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch


        print(f"✅ Loaded checkpoint from {best_model_path}")
        print(f"Resuming from epoch {start_epoch} with best validation loss {best_val_loss:.3f}")


    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Starting training from scratch.")
        # Reset if loading failed
        best_val_loss = float('inf')
        start_epoch = 0


# === Training Loop ===
num_epochs = 15 # Total number of epochs to run
epochs_to_run = num_epochs - start_epoch # Calculate remaining epochs

if epochs_to_run <= 0:
    print("Training already completed for the specified number of epochs.")
else:
    best_val_mae = float("inf")    # best validation MAE (re-initialize if not loaded from checkpoint)


    for epoch in tqdm(range(start_epoch, start_epoch + epochs_to_run), desc="Training Progress"):
        model.train()
        train_losses = []
        train_mae = []

        # Retrieve three images, labels, and dish_ids from the loader
        for images_rgb, images_heat, images_depth, labels, _ in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1} Training"):
            # Move all tensors to the device
            images_rgb = images_rgb.to(device)
            images_heat = images_heat.to(device)
            images_depth = images_depth.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Pass the three image tensors to the model's forward method
            outputs = model(images_rgb, images_heat, images_depth)

            # Ensure outputs and labels have compatible shapes for loss calculation
            # Assuming the model outputs [batch_size, 1] and labels are [batch_size, 1]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_mae.append(mean_absolute_error(labels.cpu().numpy(), outputs.detach().cpu().numpy()))

        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_mae)

        # --- Validation ---
        model.eval()
        running_loss = 0.0
        val_mae = []
        val_mape = []

        with torch.no_grad():
            # Retrieve three images, labels, and dish_ids from the loader
            for images_rgb, images_heat, images_depth, labels, _ in tqdm(val_loader, leave=False, desc=f"Epoch {epoch+1} Validation"):
                # Move all tensors to the device
                images_rgb = images_rgb.to(device)
                images_heat = images_heat.to(device)
                images_depth = images_depth.to(device)
                labels = labels.to(device)

                # Pass the three image tensors to the model's forward method
                outputs = model(images_rgb, images_heat, images_depth)

                # Ensure outputs and labels have compatible shapes for loss calculation
                # Assuming the model outputs [batch_size, 1] and labels are [batch_size, 1]
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images_rgb.size(0) # Use size(0) of any image tensor


                # Metric calculations
                val_mae.append(mean_absolute_error(labels.cpu().numpy(), outputs.detach().cpu().numpy()))

                # Calculate MAPE, handling potential division by zero if label is 0
                absolute_error = np.abs(labels.cpu().numpy() - outputs.detach().cpu().numpy())
                # Add a small epsilon to avoid division by zero for MAPE calculation
                percentage_error = absolute_error / (labels.cpu().numpy() + 1e-6)
                val_mape.append(np.mean(percentage_error))


        avg_val_loss = running_loss / len(val_loader.dataset)
        avg_val_mae = np.mean(val_mae)
        avg_val_mape = np.mean(val_mape)

        print(
            f"\nEpoch [{epoch+1}/{start_epoch+num_epochs}] " # Corrected epoch display
            f"| Train Loss (MSE): {avg_train_loss:.3f} | Train MAE: {avg_train_mae:.2f} kcal "
            f"|| Val Loss (MSE): {avg_val_loss:.3f} | Val MAE: {avg_val_mae:.2f} kcal | Val MAPE: {avg_val_mape:.2f}%"
        )

        # --- Save best checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the checkpoint including model state, optimizer state, best loss, and epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, best_model_path)
            print(f"✅ Best model checkpoint saved to {best_model_path} (Val Loss: {best_val_loss:.3f})")


        scheduler.step()
