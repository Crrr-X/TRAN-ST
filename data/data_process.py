# -*- encoding: utf-8 -*-
import os
import random
from osgeo import gdal


class DataProcess():
    """a class for pre-processing data containing some methods
    """

    def __init__(self) -> None:
        pass

    def crop(self, input_path: str, output_path: str): 
        """crop complete images to some patches

        Args:
            input_path (str): complete image path.
            output_path (str): output path of patches.
        """
        # read domain tiff data
        dataset = gdal.Open(input_path)  
        img_array = dataset.ReadAsArray()
        proj = dataset.GetProjection()
        transf = (0.0, 2.0, 0.0, 0.0, 0.0, -2.0)
        driver = gdal.GetDriverByName("GTiff")

        # target area
        hr_array = img_array[7680:, 15360:30720]
        hr = driver.Create(output_path, 15360, 7680, 1, gdal.GDT_Float32)
        hr.SetGeoTransform(transf)
        hr.SetProjection(proj)
        hr.GetRasterBand(1).WriteArray(hr_array)
        del hr

    def tilesGenerate(self, input_path: str, mode: str, out_dir: str): 
        """crop image to tiles

        Args:
            input_path (str): img path.
            mode (str): crop size (HR:192, LR:96).
            out_dir (str): output directory of tiles.
        """
        dataset = gdal.Open(input_path)  
        img_array = dataset.ReadAsArray()
        proj = dataset.GetProjection()
        transf = dataset.GetGeoTransform()
        driver = gdal.GetDriverByName("GTiff")

        # save path
        if not os.path.exists(os.path.join(out_dir, mode)):
            os.makedirs(os.path.join(out_dir, mode))
        save_dir = os.path.join(out_dir, mode)
        height, width = img_array.shape
        index, crop_size = 1, 192 if mode == "HR" else 96
        prefix = "Dem_" + str(crop_size) + "_" + str(crop_size) + "_"
        for h in range(0, height, crop_size):
            for w in range(0, width, crop_size):
                curr_index = str(index).zfill(5)
                index += 1
                curr_img = img_array[h:h+crop_size, w:w+crop_size]
                img = driver.Create(os.path.join(
                    save_dir, prefix+curr_index+".tif"), crop_size, crop_size, 1, gdal.GDT_Float32)
                curr_transf = (transf[0]+w, transf[1], 0.0, transf[3]-h, 0.0, -transf[1])
                img.SetGeoTransform(curr_transf)
                img.SetProjection(proj)
                img.GetRasterBand(1).WriteArray(curr_img)
                del img

    def pairGenerate(self, scale: int, base_dir: str, seed: int = 19):
        random.seed(seed)

        hr_dir = os.path.join(base_dir, 'HR')
        name_lists_hr = sorted(
            [name for name in os.listdir(hr_dir) if name.endswith(".tif")])
        # name_lists_hr = random.sample(name_lists_hr, 5000)
        lr_dir = os.path.join(base_dir, 'LR')
        # name_lists_lr = sorted(
        #     [name for name in os.listdir(lr_dir) if name.endswith(".TIF")])
        total_num = len(name_lists_hr)
        hr_lists = random.sample(name_lists_hr, total_num)
        replace_size = 192//scale
        lr_lists = [h.replace('Dem_192_192_', 'Dem_'+str(replace_size) +
                              '_'+str(replace_size)+'_') for h in hr_lists]
        feat_dir = os.path.join(base_dir, 'Feats')
        feats = []
        for f in os.scandir(feat_dir): 
            if not f.is_dir():
                continue
            cur_feat = [os.path.join(f.path, i) for i in hr_lists] 
            feats.append(cur_feat)
        dataset_list = []
        for hr, lr, *sub_feats in zip(hr_lists, lr_lists, *feats):
            dataset_list.append(os.path.join(hr_dir, hr) + " " +
                                os.path.join(lr_dir, lr) + " " + " ".join(sub_feats) + "\n")
        train_list = random.sample(dataset_list, int(0.8*total_num))
        val_list = [i for i in dataset_list if i not in train_list]
        print("num of Train_dataset:", len(train_list))
        print("num of Val_dataset:", len(val_list))

        with open(os.path.join(base_dir, "train_" + str(scale) + "x.txt"), "w") as f:
            for name in train_list:
                f.write(name)

        with open(os.path.join(base_dir, "val_" + str(scale) + "x.txt"), "w") as f:
            for name in val_list:
                f.write(name)


if __name__ == "__main__":
    dp = DataProcess()
    generate_path = "The file address where your data is stored"
    dp.pairGenerate(4, generate_path, 19) 
