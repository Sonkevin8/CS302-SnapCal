
class NutritionDataset(data.Dataset):
    def __init__(self, csv_file, img_dir, transform_rgb=None, mono_transform=None):
        self.labels = pd.read_csv(csv_file)

        # Drop rows with NaN in 'original_dish_id' or 'calories'
        self.labels = self.labels.dropna(subset=['original_dish_id', 'calories'])

        # Filter for positive calories
        self.labels = self.labels[self.labels['calories'] > 0]

        self.img_dir = img_dir
        self.transform_rgb = transform_rgb
        self.mono_transform = mono_transform

        # Modify filter to check for _rgbd.png and _gray.png
        initial_count = len(self.labels)
        self.labels = self.labels[self.labels['original_dish_id'].apply(
            lambda x: os.path.exists(os.path.join(self.img_dir, f"{int(x)}_rgbd.png")) and
                      os.path.exists(os.path.join(self.img_dir, f"{int(x)}_gray.png"))
        )
        ].reset_index(drop=True)

        filtered_count = len(self.labels)
        print(f"Labels shape after filtering (NaN, positive calories, and _rgbd.png + _gray.png existence): {self.labels.shape}")
        print(f"Filtered out {initial_count - filtered_count} samples due to missing files during initial check.")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        dish_id = int(row['original_dish_id'])

        # Construct image names based on available files
        img_name_rgbd = f"{dish_id}_rgbd.png"
        img_name_gray = f"{dish_id}_gray.png"
        img_name_rgb = f"{dish_id}_rgb.png" # Check if separate rgb exists

        # Construct image paths
        img_path_rgbd = os.path.join(self.img_dir, img_name_rgbd)
        img_path_gray = os.path.join(self.img_dir, img_name_gray)
        img_path_rgb = os.path.join(self.img_dir, img_name_rgb)

        # Return dummy data if there's an error loading any image
        dummy_rgb = torch.zeros(3, 448, 448, dtype=torch.float32)
        dummy_mono = torch.zeros(1, 448, 448, dtype=torch.float32)
        dummy_label = torch.tensor([0.0], dtype=torch.float32)
        dummy_dish_id = torch.tensor(0)


        try:
            # Load images and normalize to [0, 1]
            # Prioritize separate _rgb.png if it exists, otherwise extract from _rgbd.png
            try:
                if os.path.exists(img_path_rgb):
                     image_rgb = decode_image(img_path_rgb).float() / 255.0
                else:
                     # Assuming first 3 channels of _rgbd.png are RGB
                     image_rgbd_full = decode_image(img_path_rgbd).float() / 255.0
                     image_rgb = image_rgbd_full[:3, :, :] # Take only RGB channels
            except Exception as e:
                print(f"Warning: Error loading or processing RGB image for Dish ID {dish_id} from {img_path_rgb} or {img_path_rgbd}: {e}")
                return dummy_rgb, dummy_mono, dummy_mono, dummy_label, dummy_dish_id


            try:
                image_heat = decode_image(img_path_gray).float() / 255.0 # Load gray as heat
            except Exception as e:
                 print(f"Warning: Error loading or processing Heat image for Dish ID {dish_id} from {img_path_gray}: {e}")
                 return dummy_rgb, dummy_mono, dummy_mono, dummy_label, dummy_dish_id


            try:
                image_depth = decode_image(img_path_rgbd).float() / 255.0 # Load rgbd for depth
            except Exception as e:
                 print(f"Warning: Error loading or processing Depth image for Dish ID {dish_id} from {img_path_rgbd}: {e}")
                 return dummy_rgb, dummy_mono, dummy_mono, dummy_label, dummy_dish_id


            # Convert heat and depth to single channel if they are loaded as 3 channels
            if image_heat.shape[0] == 3:
                # Use Grayscale transform to convert to 1 channel
                image_heat = Grayscale(num_output_channels=1)(image_heat)
            if image_depth.shape[0] == 3:
                 # Assuming depth is encoded in intensity for 3-channel gray, take one channel
                 image_depth = Grayscale(num_output_channels=1)(image_depth)
            elif image_depth.shape[0] > 3:
                 # Assuming depth is the 4th channel in rgbd
                 image_depth = image_depth[3:4, :, :] # Take only the 4th channel
            elif image_depth.shape[0] == 1:
                 # It's already 1 channel, do nothing
                 pass
            else:
                 print(f"Warning: Unexpected number of channels ({image_depth.shape[0]}) for depth image for Dish ID {dish_id}.")
                 return dummy_rgb, dummy_mono, dummy_mono, dummy_label, dummy_dish_id


            # Apply transform
            if self.transform_rgb:
                image_rgb = self.transform_rgb(image_rgb)

            if self.mono_transform:
                image_heat = self.mono_transform(image_heat)
                image_depth = self.mono_transform(image_depth)


            # Label (calories)
            label = torch.tensor([row['calories']], dtype=torch.float32)

            # Return RGB, Heat, Depth, label, and dish_id
            return image_rgb, image_heat, image_depth, label, dish_id

        except Exception as e:
            print(f"Warning: An unexpected error occurred for Dish ID {dish_id}: {e}")
            return dummy_rgb, dummy_mono, dummy_mono, dummy_label, dummy_dish_id
\