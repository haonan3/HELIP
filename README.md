# Boosting Visual-Language Models by Exploiting Hard Pairs


## Step 1: Downloading and Organizing the Dataset

### Example: Conceptual Captions 3M (CC3M)

1. **Download the Dataset:**
   - Visit the [Conceptual Captions Download Page](https://ai.google.com/research/ConceptualCaptions/download).
   - Click on the download link to obtain the 500MB `.tsv` file.

2. **Prepare the Dataset File:**
   - Add column names to the `.tsv` file using the following command:
     ```bash
     sed -i '1s/^/caption\turl\n/' cc3m.tsv
     ```

3. **Download Images Using `img2dataset`:**
   - Create a directory for the CC3M dataset and navigate into it:
     ```bash
     mkdir /YOUR_DOWNLOAD_PATH/cc3m
     cd /YOUR_DOWNLOAD_PATH/cc3m
     ```
   - Use the `img2dataset` tool to download images according to the URLs and captions listed in the `.tsv` file:
     ```bash
     img2dataset --url_list cc3m.tsv --input_format "tsv" \
       --url_col "url" --caption_col "caption" --output_format webdataset \
       --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256 \
       --enable_wandb True
     ```
   - Replace `/YOUR_DOWNLOAD_PATH` with the actual path where you wish to download and store the dataset.

For the instructions about other dataset's downloading, please check [img2dataset](https://github.com/rom1504/img2dataset/tree/main/dataset_examples).

## Step 2: Cooking Hard Pairs

Run the `hard_pair_mining.py` script to generate hard pairs for the dataset:

```bash
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python hard_pair_mining.py --dataset cc3m --save_path /YOUR_SAVE_PATH --topk YOUR_K
```

- **Output Files:**
  - The mapping from image URLs to index files will be saved in `/YOUR_SAVE_PATH/url_index`.
  - The hard pairs will be stored in `/YOUR_SAVE_PATH/cc3m_hard_sample.csv`.
  - The hard pair dict will be stored in `/YOUR_SAVE_PATH/cc3m_hard_sample_dict.json`.
- In the `cc3m_hard_sample.csv` file, each line begins with the target pair's index, followed by the indices of the top `YOUR_K` hard pairs.

Replace `/YOUR_SAVE_PATH` with the desired save location and `YOUR_K` with the specific number of hard pairs you wish to identify.

## Step 3: Boosting Existing CLIP with HELIP

HELIP seamlessly integrates with the existing CLIP model training pipeline. As a primary example, we utilize the widely adopted [OpenCLIP](https://github.com/mlfoundations/open_clip) framework. To incorporate HELIP, you may either clone OpenCLIP and substitute the `src/training/data.py` and `src/training/params.py` files with our modified versions or directly use the adapted code available in the `src` folder provided by us.


Before training, convert the webdataset data into csv format by using `wbs_to_csv.py`. Then training with hard samples:

```bash
CUDA_VISIBLE_DEVIES=0,1,2,3,4,5,6,7  torchrun --nproc_per_node 8 -m training.main \
--train-data '/YOUR_DOWNLOAD_PATH/cc3m/cc3m.csv' \
--train-num-samples 3000000 \
--dataset-type csv \
--batch-size 420 \
--precision amp \
--workers 4 \
--csv-img-key filepath \
--csv-separator , \
--imagenet-val your/path/to/imagenet/imagenet/val \
--use-hard \
--hard-dict-dir '/YOUR_SAVE_PATH/cc3m_hard_sample.csv' \
--zeroshot-frequency 1 \
--pretrained YOUR_PRETRAINED_MODEL
```

---

Ensure to replace placeholders like `/YOUR_DOWNLOAD_PATH`, `/YOUR_SAVE_PATH`, and `YOUR_K` with actual values based on your setup and requirements.
