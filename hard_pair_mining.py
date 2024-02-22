
import os
import torch
import webdataset as wds
from torchvision import transforms
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from tqdm import tqdm, trange
import numpy as np
import torch.utils.data as tdata
import csv
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
device = "cuda" if torch.cuda.is_available() else "cpu"
import time


DATA_DIR = {'cc3m': '/YOUR_DOWNLOAD_PATH/cc3m',
            'cc12m': '/YOUR_DOWNLOAD_PATH/cc12m',
            'yfcc15m': '/YOUR_DOWNLOAD_PATH/yfcc15m'}


def load_encoders(device):        
    text_model = SentenceTransformer('all-mpnet-base-v2').to(device)
    image_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc').to(device)
    model = CombinedModel(image_model=image_model, text_model=text_model)
    return model


def obtain_dataset_path(dataset, start_tar, end_tar):
    return os.path.join(DATA_DIR[dataset], '{' + f'{start_tar:05d}'  + '..' + f'{end_tar:05d}' + '}.tar')



def save_embedding(args, image_embedding_cache, text_embedding_cache, index_cache, pt_idx):
    folder_path = os.path.join(args.save_path, 'precomputed_emb', '{}_emb'.format(args.dataset))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    image_embedding_cache = torch.cat(image_embedding_cache, dim=0)
    text_embedding_cache = torch.cat(text_embedding_cache, dim=0)
    index_cache = torch.cat(index_cache, dim=0)
    torch.save(image_embedding_cache, '{}/{}_{}_image_embed.pt'.format(folder_path, args.dataset, pt_idx))
    torch.save(text_embedding_cache,  '{}/{}_{}_text_embed.pt'.format(folder_path, args.dataset, pt_idx))
    torch.save(index_cache, '{}/{}_{}_index.pt'.format(folder_path, args.dataset, pt_idx))
    print('Saved {}-th batch'.format(pt_idx))
    pt_idx += 1
    return pt_idx



class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


class CombinedModel(nn.Module):
    def __init__(self, model=None, image_model=None, text_model=None):
        super().__init__()
        self.model = model
        self.image_model = image_model
        self.text_model = text_model

    def eval(self):
        if self.model is None:
            self.image_model.eval()
            self.text_model.eval()
        else:
            self.model.eval()

    def train(self, **kwargs):
        if self.model is None:
            self.image_model.train()
            self.text_model.train()
        else:
            self.model.train()

    def encode_image(self, x):
        if self.model is None:
            return self.image_model(x)
        else:
            return self.model.encode_image(x)

    def encode_text(self, x):
        if self.model is None:
            # return self.text_model(x)
            return self.text_model.encode(x, convert_to_tensor=True, convert_to_numpy=False, normalize_embeddings=False)
        else:
            return self.model.encode_text(x)



def cook_url_to_index(args):
    def identity(x):
        return x
    
    def preprocess(sample):
        json = sample
        try:
            url = json[0]['url']
        except:
            url = ''
        return url

    folder_path =  os.path.join(args.save_path, 'url_index')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    url_id_dict = {}
    dataset_path = obtain_dataset_path(args.dataset, args.start_tar, args.end_tar)
    dataset = wds.WebDataset(dataset_path).decode("pil").to_tuple("json").map_tuple(identity)
    dataset = dataset.map(preprocess)
    loader = torch.utils.data.DataLoader(dataset, num_workers=128, batch_size=10000)
    
    for url in tqdm(loader):
        for url_item in url:
            if url_item not in url_id_dict:
                url_id_dict[url_item] = len(url_id_dict)

    filename = '{}/{}_url_index.json'.format(folder_path, args.dataset)
    with open(filename, 'w') as file:
        json.dump(url_id_dict, file)
    print('url_id_dict saved. Length: {}'.format(len(url_id_dict)))
    return url_id_dict


def cook_representation(args, url_index_dict):
    def identity(x):
        return x

    def preprocess(sample):
        image, json = sample
        try:
            caption = json['caption']
            url = json['url']
            _index = torch.LongTensor([url_index_dict[url]])
        except:
            caption = ''
            _index = torch.LongTensor([-1]) 
        return image, caption, _index

    transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    MaybeToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    model = load_encoders(device)
    dataset_path = obtain_dataset_path(args.dataset, args.start_tar, args.end_tar)
    dataset = wds.WebDataset(dataset_path).decode("pil").to_tuple("jpg;png", "json").map_tuple(transform, identity)
    dataset = dataset.map(preprocess)
    loader = torch.utils.data.DataLoader(dataset, num_workers=32, batch_size=5000) # 10,000
    image_embedding_cache = []
    text_embedding_cache = []
    index_cache = []

    aggregator = 0
    pt_idx = 0
    with torch.no_grad():
        t = time.time()
        for image, caption, index in tqdm(loader, total=(args.end_tar-args.start_tar+1)*(10000//loader.batch_size)):
            image = image.to(device)
            image_embedding = model.encode_image(image)
            text_embedding = model.encode_text(caption)
            image_embedding_cache.append(image_embedding.cpu())
            text_embedding_cache.append(text_embedding.cpu())
            index_cache.append(index)
            aggregator += 1
            if aggregator % 50 == 0:
                print('Time: {:.2f} s'.format(time.time() - t))
                pt_idx = save_embedding(args, image_embedding_cache, text_embedding_cache, index_cache, pt_idx)
                text_embedding_cache, image_embedding_cache, index_cache = [], [], []
                torch.cuda.empty_cache()
        pt_idx = save_embedding(args, image_embedding_cache, text_embedding_cache, index_cache, pt_idx)
    print("Finished, dataset_path: {}".format(dataset_path))


def load_preprocess(args):
    img_emb_all, txt_emb_all, idx_all = None, None, None
    saved_folder = os.path.join(args.save_path, 'precomputed_emb', '{}_emb'.format(args.dataset))

    files = os.listdir(saved_folder)
    img_emb_paths = []
    cap_emb_paths = []
    idx_emb_paths = []
    for _file in files:
        file_path = os.path.join(saved_folder, _file)
        parse_info = _file.split('_')
        if 'image' in _file:
            img_emb_paths.append([int(parse_info[1]), parse_info[2], file_path])
        elif 'text' in _file:
            cap_emb_paths.append([int(parse_info[1]), parse_info[2], file_path])
        elif 'index' in _file:
            idx_emb_paths.append([int(parse_info[1]), parse_info[2], file_path])
        else:
            raise ValueError('Unknown file type: {}'.format(_file))
    img_emb_paths = sorted(img_emb_paths, key=lambda x: (x[0]))
    cap_emb_paths = sorted(cap_emb_paths, key=lambda x: (x[0]))
    idx_emb_paths = sorted(idx_emb_paths, key=lambda x: (x[0]))
    assert len(img_emb_paths) == len(cap_emb_paths) == len(idx_emb_paths)
    img_emb = torch.cat([torch.load(f[2]) for f in img_emb_paths])
    txt_emb = torch.cat([torch.load(f[2]) for f in cap_emb_paths])
    idx = torch.cat([torch.load(f[2]) for f in idx_emb_paths])

    img_emb_all = img_emb if img_emb_all is None else torch.cat([img_emb_all, img_emb])
    txt_emb_all = txt_emb if txt_emb_all is None else torch.cat([txt_emb_all, txt_emb])
    idx_all = idx if idx_all is None else torch.cat([idx_all, idx])
    return img_emb_all, txt_emb_all, idx_all


def filter_out_func(batch_rr_input, threshold, mask_self=False):
    if mask_self:
        batch_rr_input[batch_rr_input > 0.995] = -1 # the data itself
    batch_rr_input[batch_rr_input > threshold] = 0 # zero out to filter out potential duplication
    return batch_rr_input


def multi_gpu_HPM(args, new_idx_dataloader, base_txt_feat, base_img_feat):
    ngpu = torch.cuda.device_count()
    print("\nNumber of GPU in used: {} \n.".format(ngpu))
    assert ngpu > 1
    sample_num = base_txt_feat.shape[0]

    result_cache = []
    image_base = []
    text_base = []
    sampled_idx_list = []
    round_id = 0
    for idxs in tqdm(new_idx_dataloader):
        # renew base feat every 300000 samples
        bz = new_idx_dataloader.batch_size
        current_round_id = bz * len(result_cache) // 300000 
        if len(result_cache) == 0 or current_round_id > round_id:
            round_id = current_round_id
            # 1.sample base feat
            if args.approx < sample_num:
                sampled_idx = np.random.choice(sample_num, args.approx, replace=False).astype(int)
                sampled_idx = np.sort(sampled_idx)
                _base_txt_feat = base_txt_feat[sampled_idx]
                _base_img_feat = base_img_feat[sampled_idx]
            else:
                _base_txt_feat = base_txt_feat
                _base_img_feat = base_img_feat
                sampled_idx = np.arange(sample_num).astype(int)
            del image_base, text_base, sampled_idx_list
            torch.cuda.empty_cache()

            # 2. move base feat to multiple gpus
            sampled_idx_list = []
            for i in range(ngpu):
                sampled_idx_list.append( torch.LongTensor(sampled_idx).to('cuda:' + str(i)) )
            image_base = []
            text_base = []
            partition_bz = args.approx // ngpu + 1
            start, end = 0, 0
            chunk_pad_num_list = []
            for gpu_i in trange(ngpu):
                end += partition_bz
                chunk_pad_num_list.append(start)
                if args.mining_method in ["HPM", "text"]:
                    text_base.append(F.normalize(_base_txt_feat[start:end].to('cuda:' + str(gpu_i)), p=2, dim=-1).T)
                if args.mining_method in ["HPM", "image"]:
                    image_base.append(F.normalize(_base_img_feat[start:end].to('cuda:' + str(gpu_i)), p=2, dim=-1).T)
                start = end

        key_txt_cache, key_img_cache = [], []
        if args.mining_method in ["HPM", "text"]:
            key_txt_feat = F.normalize(base_txt_feat[idxs].to('cuda:0'), p=2, dim=-1)
            key_txt_cache.append(key_txt_feat)
        if args.mining_method in ["HPM", "image"]:
            key_img_feat = F.normalize(base_img_feat[idxs].to('cuda:0'), p=2, dim=-1)
            key_img_cache.append(key_img_feat)

        for i in range(ngpu):
            if i != 0:
                if args.mining_method in ["HPM", "text"]:
                    key_txt_cache.append(key_txt_feat.to('cuda:' + str(i)))
                if args.mining_method in ["HPM", "image"]:
                    key_img_cache.append(key_img_feat.to('cuda:' + str(i)))

        topk_val_list, topk_idx_list = [], []
        for i in range(ngpu):
            if args.mining_method == 'HPM':
                batch_rr_text = torch.matmul(key_txt_cache[i], text_base[i])
                batch_rr_image = torch.matmul(key_img_cache[i], image_base[i])
                threshold = 0.98
                batch_rr_text = filter_out_func(batch_rr_text, threshold, mask_self=True)
                batch_rr_image = filter_out_func(batch_rr_image, threshold)
                batch_result_i = batch_rr_image * batch_rr_text
            elif args.mining_method == 'text':
                batch_result_i = torch.matmul(key_txt_cache[i], text_base[i])
                threshold = 0.98
                batch_result_i = filter_out_func(batch_result_i, threshold)
            else:
                batch_result_i = torch.matmul(key_img_cache[i], image_base[i])
                threshold = 0.98
                batch_result_i = filter_out_func(batch_result_i, threshold)

            _topk_val, _topk_idx = batch_result_i.topk(args.topk, 1, True, True)
            topk_val_list.append(_topk_val)
            assert _topk_idx.device.index == i
            topk_idx = sampled_idx_list[_topk_idx.device.index][_topk_idx + chunk_pad_num_list[i]]
            topk_idx_list.append(topk_idx)

        batch_result_val, batch_result_idx = -1 * torch.ones(idxs.shape[0], 1 + args.topk * ngpu).float(), \
                    -1 * torch.ones(idxs.shape[0], 1 + args.topk * ngpu).long()
        start_inner, end_inner = 0, 0
        batch_result_val[:, 0] = 1
        batch_result_idx[:, 0] = idxs
        for i in range(ngpu):
            end_inner += topk_val_list[i].shape[1]
            batch_result_val[:, 1+start_inner:1+end_inner] = topk_val_list[i].cpu()
            batch_result_idx[:, 1+start_inner:1+end_inner] = topk_idx_list[i].cpu()
            start_inner = end_inner

        reselect_topk = batch_result_val.topk(args.topk + 1, 1, True, True)[1]
        pad = torch.range(0, (batch_result_idx.shape[0] - 1) * batch_result_idx.shape[1], batch_result_idx.shape[1]).long().unsqueeze(-1)
        assert pad.shape[0] == reselect_topk.shape[0]
        reselect_topk = (pad + reselect_topk).reshape(-1)
        result_idx = batch_result_idx.reshape(-1)[reselect_topk].reshape(batch_result_idx.shape[0], -1)
        result_cache.append(result_idx)
    hard_sample_idx = torch.cat(result_cache, dim=0).numpy()
    return hard_sample_idx



def parsers_parser():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='cc3m', help='cc3m, cc12m, yfcc15m')
    parser.add_argument('--start_tar', type=int, default=0)
    parser.add_argument('--end_tar', type=int, default=331, help="Specify the ending tar file index for dataset downloads. For example, if downloading the CC3M dataset, which is segmented into tar files named from {00000..00331}.tar, set 'end_tar' to 331. For the CC12M dataset, it is {00000..01230}.tar; for YFCC15M, it is {00000..01538}.tar.")
    parser.add_argument('--save_path', type=str, default=None, help='set the data save path')

    parser.add_argument('--topk', type=int, default=100, help='choose top k sparse vector')
    parser.add_argument('--mining_method', type=str, default='HPM', help='HPM, image, text')
    parser.add_argument('--approx', type=int, default=3000000)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parsers_parser()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    url_id_dict = cook_url_to_index(args)
    cook_representation(args, url_id_dict)

    ''' 1.Load preprocessed base '''
    img_emb, txt_emb, url_idx = load_preprocess(args)
    assert np.sum(url_idx.numpy() == -1) == 0     # check if url_idx includes -1
    new_idx_to_url_idx_dict = {}
    for _new_index , _url_index in enumerate(url_idx.squeeze(-1).tolist()):
        new_idx_to_url_idx_dict[_new_index] = int(_url_index)

    ''' 2.HPM '''
    sample_num = txt_emb.shape[0]
    bz = 600 # the size of batch size is dependent on your GPU memory
    new_idx_dataloader = tdata.DataLoader(torch.LongTensor(list(range(sample_num))), batch_size=bz, shuffle=False, drop_last=False)    
    hard_sample_idx = multi_gpu_HPM(args, new_idx_dataloader, txt_emb, img_emb)

    ''' 3.Save hard sample '''
    print(hard_sample_idx.shape)
    save_file_name = os.path.join(args.save_path, '{}_hard_sample.csv'.format(args.dataset))
    
    with open(save_file_name, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        map_func = np.vectorize(lambda x: new_idx_to_url_idx_dict.get(x, x)) 
        hard_sample_result = map_func(hard_sample_idx)
        writer.writerows(hard_sample_result)