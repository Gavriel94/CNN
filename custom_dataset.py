from torchvision.transforms import v2
from torch.utils.data import Dataset
import os
from PIL import Image

class ModelDataset(Dataset):
    """
    Enables transformations to be applied to an image and its copies
      before they are presented to the model.
    """
    def __init__(self, data):
        """
        Defines transformations and stores the data.

        Args:
            data (list[int, str]): label and its 
                corresponding image path.
        """
        
        self.data = data
        self.general_transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.Pad(padding=4, fill=(0,0,0), padding_mode='constant'),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Retrieves the label, and the image 
        Applies transformations to the image and its copies
        

        Args:
            index (int): Index to retrieve data

        Returns:
            label, image, brightened_image, darkened_image 
            (int, torch.Tensor, torch.Tensor, torch.Tensor): 
                label and transformed images.
        """
        path = os.path.join('Data/Faces Grouped/', self.data[index][1])
        label = self.data[index][0]
        image = Image.open(path).convert("RGB")
        
        #increase brightness by 50%
        b_image  = v2.functional.adjust_brightness(image, 1.5)
        brightened_image = self.general_transforms(b_image)
        
        #reduce brightness by 50%
        d_image = v2.functional.adjust_brightness(image, 0.5)
        darkened_image = self.general_transforms(d_image)
        
        image = self.general_transforms(image)
        return label, image, brightened_image, darkened_image