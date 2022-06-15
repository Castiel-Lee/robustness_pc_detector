import os
import logging
from pathlib import Path
from datetime import datetime



def filter_split_files(log: logging.Logger, split_folder : str,
                       log_files_folder: str = str(Path.home() / 'Downloads')):

    out_files = []

    for file in os.listdir(log_files_folder):

        if file.endswith(".log"):
            out_files.append(os.path.join(log_files_folder, file))

    missing_samples = []

    for out_file in out_files:

        with open(out_file) as f:

            for line in f:

                if 'ERROR' in line:

                    file_name = line.split(' ')[-3]
                    sample_string = file_name.split('/')[-1].split('.')[0].split('_')
                    sample_id = f'{sample_string[0]}_{sample_string[1]},{sample_string[2]}'
                    missing_samples.append(sample_id)

    unique_list = sorted(set(missing_samples))

    log.info(f'there are {len(unique_list)} missing VLP32 frames')

    split_files = []

    for file in os.listdir(split_folder):

        if file.endswith("_vlp32.txt"):
            split_files.append(os.path.join(split_folder, file))

    for split_file in sorted(split_files):

        with open(split_file, 'r') as file:
            filedata = file.read()

        for sample in unique_list:

            term = sample + '\n'

            if term in filedata:
                filedata = filedata.replace(term, '')
                log.info(f'{sample} removed from {Path(split_file).name}')

        with open(split_file, 'w') as file:
            file.write(filedata)

if __name__ == '__main__':

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY_H:M:S
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    this_file_location = Path(__file__).parent.resolve()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)5s  %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=this_file_location / f'log_{dt_string}.txt')
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console)

    filter_split_files(log=logger, split_folder=str(this_file_location.parent / 'splits'))


