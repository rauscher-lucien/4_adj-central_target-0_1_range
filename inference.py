import os
import sys
sys.path.append(os.path.join(".."))

import torch
import numpy as np
import tifffile
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

from model import *
from transforms import *
from utils import *
from dataset import *


def load(dir_chck, netG, epoch, optimG=[]):

    dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch), map_location=torch.device('cpu'))

    print('Loaded %dth network' % epoch)

    netG.load_state_dict(dict_net['netG'])
    optimG.load_state_dict(dict_net['optimG'])

    return netG, optimG, epoch

def main():

    #********************************************************#

    project_dir = os.path.join('Z:\\', 'members', 'Rauscher', 'projects', '4_adj-central_target-0_1_range')
    # project_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', '4_adj-central_target-0_1_range')
    data_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_data_2')
    name = 'test_data_2-test-1'
    inference_name = 'inference_150'
    load_epoch = 150


    #********************************************************#

    results_dir = os.path.join(project_dir, name, 'results')
    checkpoints_dir = os.path.join(project_dir, name, 'checkpoints')

    # Make a folder to store the inference
    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)
    
    ## Load image stack for inference
    filenames = glob.glob(os.path.join(data_dir, "*.TIFF"))
    print("Following file will be denoised:  ", filenames[0])



    # check if GPU is accessible
    if torch.cuda.is_available():
        print("\nGPU will be used.")
        device = torch.device("cuda:0")
    else:
        print("\nCPU will be used.")
        device = torch.device("cpu")

    min, max = load_min_max_params(data_dir=data_dir)
    # mean, std = load_normalization_params(data_dir=data_dir)
    
    inf_transform = transforms.Compose([
        MinMaxNormalizeInference(min, max),
        CropToMultipleOf32Inference(),
        ToTensor(),
    ])

    inv_inf_transform = transforms.Compose([
        BackTo01Range(),
        ToNumpy()
    ])

    inf_dataset = N2N4InputSliceInferenceDataset(
        data_dir,
        transform=inf_transform
    )

    batch_size = 8
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    
    netG = NewUNet()
    # init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=0)
    paramsG = netG.parameters()
    optimG = torch.optim.Adam(paramsG, lr=1e-3, betas=(0.5, 0.999))
    netG, optimG, st_epoch = load(checkpoints_dir, netG, load_epoch, optimG)

    num_inf = len(inf_dataset)
    num_batch = int((num_inf / batch_size) + ((num_inf % batch_size) != 0))

    print("starting inference")
    output_images = []  # List to collect output images

    with torch.no_grad():
        netG.eval()

        for batch, data in enumerate(inf_loader):
            input_img = data.to(device)  # Assuming data is already a tensor of the right shape

            output_img = netG(input_img)
            output_img_np = inv_inf_transform(output_img)  # Convert output tensors to numpy format for saving

            for img in output_img_np:
                output_images.append(img)

            print('BATCH %04d/%04d' % (batch, len(inf_loader)))

    # Clip output images to the 0-1 range
    output_images_clipped = [np.clip(img, 0, 1) for img in output_images]
    
    # Stack and save output images
    output_stack = np.stack(output_images_clipped, axis=0)
    tifffile.imwrite(os.path.join(inference_folder, 'output_stack.TIFF'), output_stack)

    print("TIFF stacks created successfully.")

if __name__ == '__main__':
    main()



#     print("starting inference")
#     with torch.no_grad():

#         netG.eval()

#         for batch, data in enumerate(inf_loader):

#             input_img = data[0].to(device)
#             output_img = netG(input_img)

#             input_img = inv_inf_transform(input_img)[..., 0]
#             output_img = inv_inf_transform(output_img)[..., 0]

#             # input_img = np.clip(input_img, 0, 1)
#             # output_img = np.clip(output_img, 0, 1)

#             for j in range(0, batch_size):
#                 name1 = batch
#                 name2 = j
#                 fileset = {'name': name,
#                             'input': "%04d-%04d-input.png" % (name1, name2),
#                             'output': "%04d-%04d-output.png" % (name1, name2),
#                             'target': "%04d-%04d-label.png" % (name1, name2)}

#                 input_img_1 = np.squeeze(input_img[j, :, :])
#                 output_img_1 = np.squeeze(output_img[j, :, :])

#                 input_img_1 = (input_img_1 * 255).astype(np.uint8)
#                 output_img_1 = (output_img_1 * 255).astype(np.uint8)

#                 input_img_path = os.path.join(inference_folder, fileset['input'])
#                 output_img_path = os.path.join(inference_folder, fileset['output'])

#                 # Image.fromarray(input_img_1).save(input_img_path)
#                 Image.fromarray(output_img_1).save(output_img_path)


# if __name__ == '__main__':
#     main()

