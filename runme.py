import os
from PIL import Image
import torch
import numpy as np
import cv2
from DPR.utils.utils_SH import *

os.environ['CUDA_VISIBLE_DEVICES'] = input("Enter GPU number >> ")
device = "cuda"

from GeomConsistentFR.RelightNet import RelightNet
import imageio
from tqdm import tqdm, trange

model = RelightNet()
model.load_state_dict(torch.load('GeomConsistentFR/model_lighting_transfer/model_epoch106.pth'))
model = model.float()
model = model.cuda()
model.eval()

epoch = 200
intrinsic_matrix = np.zeros((1, 3, 3))
intrinsic_matrix[:, 0, 0] = 700.0
intrinsic_matrix[:, 1, 1] = 700.0
intrinsic_matrix[:, 2, 2] = 1.0
intrinsic_matrix[:, 0, 2] = model.img_width/2.0
intrinsic_matrix[:, 1, 2] = model.img_height/2.0
intrinsic_matrix = torch.from_numpy(intrinsic_matrix)

curr_mask_fill_nose = np.zeros((256, 256, 1))
curr_mask_fill_nose.fill(1)
curr_mask_fill_nose = torch.from_numpy(curr_mask_fill_nose)

curr_mask_fill_nose_3_channels = np.zeros((model.img_height, model.img_width, 3))
curr_mask_fill_nose_3_channels[:, :, 0] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))
curr_mask_fill_nose_3_channels[:, :, 1] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))
curr_mask_fill_nose_3_channels[:, :, 2] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))

def decompose(image_path):
    curr_input_image = torch.reshape(torch.from_numpy(imageio.imread(image_path)/255.0), (1, 256, 256, 3)) 
    curr_reference_image = curr_input_image
    curr_training_lighting = torch.from_numpy(np.zeros((model.batch_size, 4)))
    
    albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light \
        = model(curr_reference_image.float().cuda(), epoch, intrinsic_matrix.cuda(), curr_mask_fill_nose.cuda(), torch.reshape(curr_training_lighting[:, 1:4].float().cuda(), (model.batch_size, 3, 1, 1)), torch.reshape(curr_training_lighting[:, 0].float().cuda(), (model.batch_size, 1, 1)))
        
    albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light \
        = model(curr_input_image.float().cuda(), epoch, intrinsic_matrix.cuda(), curr_mask_fill_nose.cuda(), torch.reshape(estimated_unit_light_direction.float().cuda(), (model.batch_size, 3, 1, 1)), torch.reshape(estimated_ambient_light.float().cuda(), (model.batch_size, 1, 1)))
        
    rendered_images = rendered_images.permute(0, 2, 3, 1)
    rendered_images = rendered_images.cpu().detach().numpy()
    albedo = albedo.permute(0, 2, 3, 1)
    albedo = albedo.cpu().detach().numpy()
    depth = depth.permute(0, 2, 3, 1)
    depth = depth.cpu().detach().numpy()
    depth = -depth
    depth = (depth-np.amin(depth))/(np.amax(depth)-np.amin(depth))
        
    final_shading = final_shading.cpu().detach().numpy()
        
    surface_normals = surface_normals.permute(0, 2, 3, 1)
    surface_normals = surface_normals.cpu().detach().numpy()
    surface_normals = 255.0*(surface_normals+1.0)/2.0

    input_image = curr_input_image[0].detach().cpu().numpy()*255.0
    input_image = input_image[:, :, ::-1]
    rendered_image = 255.0*rendered_images[0, :, :, ::-1]*curr_mask_fill_nose_3_channels
            
    input_image[curr_mask_fill_nose_3_channels > 0] = rendered_image[curr_mask_fill_nose_3_channels > 0]

    cv2.imwrite(image_path.split('.jpg')[0] + '_rendered_image.png', input_image)
    cv2.imwrite(image_path.split('.jpg')[0] + '_shadow_mask.png', 255.0*shadow_mask_weights[0, :, :].cpu().detach().numpy()*np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width)))
    cv2.imwrite(image_path.split('.jpg')[0] + '_albedo.png', 255.0*albedo[0, :, :, ::-1]*curr_mask_fill_nose_3_channels)
    cv2.imwrite(image_path.split('.jpg')[0] + '_depth.png', 255.0*depth[0, :, :, :]*curr_mask_fill_nose.numpy())
    cv2.imwrite(image_path.split('.jpg')[0] + '_shading.png', 255.0*final_shading[0, :, :]*np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width)))
    cv2.imwrite(image_path.split('.jpg')[0] + '_surface_normals.png', surface_normals[0, :, :, ::-1]*curr_mask_fill_nose_3_channels)


if __name__ == "__main__":
    train_data_path = input("Input name of the directory containing images >> ")
    
    # make directory for decomposed images
    os.makedirs(f"./{train_data_path}_decomposed", exist_ok=True)

    # for every subdirectory under "train_data_face" directory
    # for every image in the subdirectory
    for dirs in tqdm(os.listdir(f"./{train_data_path}"), desc="videos", leave=True):
        try:
            os.makedirs(f"./{train_data_path}_decomposed/{dirs}", exist_ok=False)
        except:
            # check the number of inner files in the directory
            if len(os.listdir(f"./{train_data_path}_decomposed/{dirs}")) == len(os.listdir(f"./{train_data_path}/{dirs}")) * 7:
                continue
        for file in tqdm(os.listdir(f"./{train_data_path}/{dirs}"), desc="images", leave=False):
            image_path = os.path.join(f"./{train_data_path}_decomposed/{dirs}/", file)
            # resize to 256x256
            img = Image.open(os.path.join(f"./{train_data_path}/{dirs}/", file))
            img = img.resize((256, 256), Image.ANTIALIAS)
            # save with postfix '_256x256.jpg'
            img.save(image_path.split('.jpg')[0] + '_256x256.jpg')

            with torch.no_grad():
                decompose(image_path.split('.jpg')[0] + '_256x256.jpg')
