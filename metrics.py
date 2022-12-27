import os
from PIL import Image
import numpy as np
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image

def adjust_dynamic_range(data, drange_in, drange_out): 
    if drange_in != drange_out:
 
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        
        data = data * scale + bias
    return data


def cal_psnr(arr1, arr2):
    return psnr(arr1, arr2, data_range=int(np.nanmax(arr2)))


def cal_ssim(arr1, arr2, data_range):
    return ssim(arr1, arr2, win_size=5, data_range=data_range)


def cal_cc(arr1, arr2):
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()
    return np.mean(np.corrcoef(arr1, arr2, rowvar=False))


def cal_polarity_acc_BasedOnRealImg(arr1, arr2, magnetic_thres):
    num_neg1 = 0
    num_pos1 = 0
    num_neg2 = 0
    num_pos2 = 0
    num_equal = 0

    size_x, size_y = arr1.shape[0], arr1.shape[1]  # arr.shape[0] = h ; arr.shape[1] = w

    for i in range(size_x):
        for j in range(size_y):
            if arr2[i][j] < -magnetic_thres:
                arr2[i][j] = -1
                num_neg2 += 1
                if arr1[i][j] < 0:
                    arr1[i][j] = -1
                    num_neg1 += 1
                elif arr1[i][j] > 0:
                    arr1[i][j] = 1
                    num_pos1 += 1
            elif arr2[i][j] > magnetic_thres:
                arr2[i][j] = 1
                num_pos2 += 1
                if arr1[i][j] < 0:
                    arr1[i][j] = -1
                    num_neg1 += 1
                elif arr1[i][j] > 0:
                    arr1[i][j] = 1
                    num_pos1 += 1
            else:
                arr2[i][j] = np.nan
                arr1[i][j] = np.nan

    diff = arr1 - arr2
    debug = set(list(diff.flatten()[np.isfinite(diff.flatten())]))
    print(debug)

    for i in range(size_x):
        for j in range(size_y):
            if diff[i][j] == 0:
                num_equal += 1

    num1 = num_neg1 + num_pos1
    num2 = num_neg2 + num_pos2

    if num1 == num2:
        print('num1 = num2')

    acc = num_equal/num2

    return acc


def cal_val(base_path):

    psnr_polar_list = []
    ssim_polar_list = []
    cc_polar_list = []

    polarity_acc_list = []

    os.chdir(base_path)
    fake_imgs = glob.glob('*_fake_B.png')
    print(len(fake_imgs))
    for fake_img in fake_imgs:
        # print(fake_img)
        fake_path = os.path.join(base_path, fake_img)
        real_path = os.path.join(base_path, fake_img.replace('fake', 'real'))

        arr1 = np.array(Image.open(fake_path).convert('L'))
        arr2 = np.array(Image.open(real_path).convert('L'))
        
        # binning
        # im1 = Image.fromarray(arr1)
        # im2 = Image.fromarray(arr2)
        #
        h, w = arr1.shape
        # # print(h, w)
        # bin_size = 8
        # wn, hn = w // bin_size, h // bin_size
        # im1 = im1.resize((wn, hn), Image.BILINEAR)
        # im2 = im2.resize((wn, hn), Image.BILINEAR)
        #
        # arr1 = np.array(im1)
        # arr2 = np.array(im2)

        
        # clean up the black edges and keep the disk
        # h = hn
        # w = wn
        print(h, w)
        radius = 450 #/ bin_size    # 450
        center = (int(w/2), int(h/2))

        gen = np.empty((h, w), dtype=np.float32)
        for i in range(arr1.shape[0]):
            for j in range(arr1.shape[1]):
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                if dist <= radius:
                    gen[i, j] = arr1[i, j]
                else:
                    gen[i, j] = np.nan
        arr1 = gen


        obs = np.empty((h, w), dtype=np.float32)
        for i in range(arr2.shape[0]):
            for j in range(arr2.shape[1]):
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                if dist <= radius:
                    obs[i, j] = arr2[i, j]
                else:
                    obs[i, j] = np.nan
        arr2 = obs


        # test if the number and position of nan in arr1 and arr2 is the same
        if np.where(np.isnan(arr1))[0].all() == np.where(np.isnan(arr2))[0].all():
            print('true')

        # delete the np.nan in arr1_eval and arr2_eval
        arr1_eval = arr1[~np.isnan(arr1)]
        arr2_eval = arr2[~np.isnan(arr2)]
        print('arr1_eval.shape:', arr1_eval.shape)


        psnr_polar = cal_psnr(arr1_eval, arr2_eval)
        psnr_polar_list.append(psnr_polar)
        ssim_polar = cal_ssim(arr1_eval, arr2_eval, 400)
        ssim_polar_list.append(ssim_polar)
        cc_polar = cal_cc(arr1_eval, arr2_eval)
        cc_polar_list.append(cc_polar)

        arr1 = adjust_dynamic_range(arr1, [0, 255], [-200, 200])
        arr2 = adjust_dynamic_range(arr2, [0, 255], [-200, 200])

        polarity_acc = cal_polarity_acc_BasedOnRealImg(arr1, arr2, magnetic_thres=0)
        polarity_acc_list.append(polarity_acc)

        print(polarity_acc)

    np.savetxt("psnr_polar_disk.csv", psnr_polar_list, delimiter=',')
    np.savetxt("ssim_polar_disk.csv", ssim_polar_list, delimiter=',')
    np.savetxt("cc_polar_disk.csv", cc_polar_list, delimiter=",")

    np.savetxt("polarity_acc_ar_0.csv", polarity_acc_list, delimiter=',')


if __name__ == '__main__':
    print("Processing...")
    base_path = 'your_base_path'
    cal_val(base_path)
    print("Done!")


