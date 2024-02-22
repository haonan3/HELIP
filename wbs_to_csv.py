import webdataset as wds
import pandas as pd
import tarfile
from tqdm import trange
import argparse
import os

''' This is a simple script to convert a WebDataset to a CSV file. In the following, we take cc3m as an example.'''


DATA_DIR = {'cc3m': '/YOUR_DOWNLOAD_PATH/cc3m',
            'cc12m': '/YOUR_DOWNLOAD_PATH/cc12m',
            'yfcc15m': '/YOUR_DOWNLOAD_PATH/yfcc15m'}


def parsers_parser():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='cc3m', help='cc3m, cc12m, yfcc15m')
    parser.add_argument('--start_tar', type=int, default=0)
    parser.add_argument('--end_tar', type=int, default=331, help="Specify the ending tar file index for dataset downloads. For example, if downloading the CC3M dataset, which is segmented into tar files named from {00000..00331}.tar, set 'end_tar' to 331. For the CC12M dataset, it is {00000..01230}.tar; for YFCC15M, it is {00000..01538}.tar.")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parsers_parser()

    # Define a list to hold your data
    data = []
    extract_to_path = '{}/{}_csv/'.format(DATA_DIR[args.dataset], args.dataset)

    # Initialize a WebDataset reader
    dataset_path = os.path.join(DATA_DIR[args.dataset], '{' + f'{args.start_tar:05d}'  + '..' + f'{args.end_tar:05d}' + '}.tar')
    dataset = wds.WebDataset(dataset_path)
    for sample in dataset:
        image_path = extract_to_path + sample["__key__"] + ".jpg"
        data.append({"filepath": image_path, "title": sample["txt"]})
    # Convert the list to a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    csv_path = "{}/{}.csv".format(DATA_DIR[args.dataset], args.dataset)
    df.to_csv(csv_path, index=False)
    

    for i in trange(args.start_tar, args.end_tar):
        tar_file_path = '{}/%05d.tar'.format(DATA_DIR[args.dataset]) % (i)
        with tarfile.open(tar_file_path, 'r') as tar:
            tar.extractall(path=extract_to_path)
        print(f"Finish processing {tar_file_path}")


    print(f"Dataset converted and saved to {csv_path}")

