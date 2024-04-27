import os
import cv2
import torch
from torch import nn
import torchvision.models as models


class model:
    def __init__(self):
        self.checkpoint = "resnet18.pth"
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        make sure these files are in the same directory as the model.py file.
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """

        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model = torch.load(checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        image = cv2.resize(input_image, (224, 224))
        image = image / 255
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, torch.float)

        with torch.no_grad():
            score = self.model(image)
        pred_class = torch.argmax(score, 1)
        pred_class = int(pred_class)
        return pred_class

model_ = model()
model_.load('C:/Users/Lynn/Desktop/Task2/')
img = cv2.imread("C:/Users/Lynn/Desktop/2-Hypertensive Retinopathy Classification/1-Images/dataset/1/00000dea.png", 1)
res = model_.predict(img)
print(res)