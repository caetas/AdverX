import os
import json
import cv2
from config import data_raw_dir, data_processed_dir
from tqdm import tqdm

base_dir = os.path.join(data_raw_dir, 'BIMCV-COVID19')
dest_dir = os.path.join(data_processed_dir, 'BIMCV-COVID19-processed')

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

if not os.path.exists(base_dir):
    raise Exception(f"Directory {base_dir} does not exist")

sub_folders = os.listdir(base_dir)

sub_folders = [x for x in sub_folders if os.path.isdir(os.path.join(base_dir, x))]
sub_folders = [x for x in sub_folders if 'sub' in x]
n_pngs = 0
n_saved = 0

for sub in tqdm(sub_folders, desc='Processing subfolders', total=len(sub_folders)):
    sub_dir = os.path.join(base_dir, sub)
    sub_sub_folders = os.listdir(sub_dir)
    # use only folders
    sub_sub_folders = [x for x in sub_sub_folders if os.path.isdir(os.path.join(sub_dir, x))]
    for sub_sub in sub_sub_folders:
        sub_sub_dir = os.path.join(sub_dir, sub_sub)
        sub_sub_sub_folders = os.listdir(sub_sub_dir)
        # use only folders
        sub_sub_sub_folders = [x for x in sub_sub_sub_folders if os.path.isdir(os.path.join(sub_sub_dir, x))]
        for sub_sub_sub in sub_sub_sub_folders:
            sub_sub_sub_dir = os.path.join(sub_sub_dir, sub_sub_sub, 'mod-rx')
            files = os.listdir(sub_sub_sub_dir)
            # if no png exists, skip
            if not any([x.endswith('.png') for x in files]):
                continue
            else:
                # count number of pngs
                png_files = [x for x in files if x.endswith('.png')]
                # check if jsons exist
                json_files = [x for x in files if x.endswith('.json')]

                if not json_files:
                    print(f'\t\t\tNo json files found in {sub_sub_sub_dir}')
                    continue
                else:
                    n_pngs += len(png_files)
                    for json_file in json_files:
                        with open(os.path.join(sub_sub_sub_dir, json_file), 'r') as f:
                            data = json.load(f)
                        # check if key 00081090 exists
                        if '00081090' not in data:
                            machine_dir = os.path.join(dest_dir, data['00080070']['Value'][0])
                        else:
                            #if they contain "" remove them
                            data['00080070']['Value'][0] = data['00080070']['Value'][0].replace('"','')
                            data['00081090']['Value'][0] = data['00081090']['Value'][0].replace('"','')
                            machine_dir = os.path.join(dest_dir, data['00080070']['Value'][0] + ' ' +data['00081090']['Value'][0])
                        if not os.path.exists(machine_dir):
                            os.makedirs(machine_dir)
                        
                        png_name = json_file.replace('.json', '.png')
                        pat_id = data['00100020']['Value'][0]

                        save_dir = os.path.join(machine_dir, pat_id)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        file_name = f"{data['00080060']['Value'][0]}_{data['00280004']['Value'][0]}"

                        if 'acq-' in png_name:
                            if 'acq-8' in png_name:
                                for i in range(1, 9):
                                    new_png_name = png_name.replace('acq-8', 'acq-'+str(i))
                                    if not os.path.exists(os.path.join(sub_sub_sub_dir, new_png_name)):
                                        print('DOES NOT EXIST', new_png_name)
                                    if os.path.exists(os.path.join(sub_sub_sub_dir, new_png_name)):
                                        n_saved += 1
                                        index = len(os.listdir(save_dir))
                                        cv2.imwrite(os.path.join(save_dir, f'{file_name}_{index}.png'), cv2.imread(os.path.join(sub_sub_sub_dir, new_png_name), cv2.IMREAD_UNCHANGED))
                            elif 'acq-4' in png_name:
                                for i in range(1, 5):
                                    new_png_name = png_name.replace('acq-4', 'acq-'+str(i))
                                    if not os.path.exists(os.path.join(sub_sub_sub_dir, new_png_name)):
                                        print('DOES NOT EXIST', new_png_name)
                                    if os.path.exists(os.path.join(sub_sub_sub_dir, new_png_name)):
                                        n_saved += 1
                                        index = len(os.listdir(save_dir))
                                        cv2.imwrite(os.path.join(save_dir, f'{file_name}_{index}.png'), cv2.imread(os.path.join(sub_sub_sub_dir, new_png_name), cv2.IMREAD_UNCHANGED))
                            elif 'acq-3' in png_name:
                                for i in range(1, 4):
                                    new_png_name = png_name.replace('acq-3', 'acq-'+str(i))
                                    if not os.path.exists(os.path.join(sub_sub_sub_dir, new_png_name)):
                                        print('DOES NOT EXIST', new_png_name)
                                    if os.path.exists(os.path.join(sub_sub_sub_dir, new_png_name)):
                                        n_saved += 1
                                        index = len(os.listdir(save_dir))
                                        cv2.imwrite(os.path.join(save_dir, f'{file_name}_{index}.png'), cv2.imread(os.path.join(sub_sub_sub_dir, new_png_name), cv2.IMREAD_UNCHANGED))
                            elif 'acq-2' in png_name:
                                for i in range(1, 3):
                                    new_png_name = png_name.replace('acq-2', 'acq-'+str(i))
                                    if not os.path.exists(os.path.join(sub_sub_sub_dir, new_png_name)):
                                        print('DOES NOT EXIST', new_png_name)
                                    if os.path.exists(os.path.join(sub_sub_sub_dir, new_png_name)):
                                        n_saved += 1
                                        index = len(os.listdir(save_dir))
                                        cv2.imwrite(os.path.join(save_dir, f'{file_name}_{index}.png'), cv2.imread(os.path.join(sub_sub_sub_dir, new_png_name), cv2.IMREAD_UNCHANGED))
                        else:
                            n_saved += 1
                            index = len(os.listdir(save_dir))
                            cv2.imwrite(os.path.join(save_dir, f'{file_name}_{index}.png'), cv2.imread(os.path.join(sub_sub_sub_dir, png_name), cv2.IMREAD_UNCHANGED))
                        #print(data['00100020']['Value'][0], data['00080060']['Value'][0], data['00280004']['Value'][0])

print(f'Number of pngs: {n_pngs}')
print(f'Number of saved pngs: {n_saved}')