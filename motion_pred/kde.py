import numpy as np
import argparse
import os
import sys
import pickle
import csv
import torch
import time
from thop import profile
from thop import clever_format
from scipy.spatial.distance import pdist

from abc import ABC
from typing import Optional, List
import math
from torch import Tensor

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m import DatasetH36M
from motion_pred.utils.dataset_humaneva import DatasetHumanEva
from motion_pred.utils.visualization import render_animation
from models.motion_pred import *
from scipy.spatial.distance import pdist, squareform



def kde(y, y_pred):
    y, y_pred = torch.from_numpy(y).float().to(torch.device('cuda')), torch.from_numpy(y_pred).float().to(torch.device('cuda'))
    bs, sp, ts, ns, d = y_pred.shape
    kde_ll = torch.zeros((bs, ts, ns), device=y_pred.device)

    for b in range(bs):
        for t in range(ts):
            for n in range(ns):
                try:
                    kernel = GaussianKDE(y_pred[b, :, t, n, :])
                except BaseException:
                    print("b: %d - t: %d - n: %d" % (b, t, n))
                    continue
                # pred_prob = kernel(y_pred[:, b, t, :, n])
                gt_prob = kernel(y[b, None, t, n, :])
                kde_ll[b, t, n] = gt_prob
    # mean_kde_ll = torch.mean(kde_ll)
    mean_kde_ll = torch.mean(torch.mean(kde_ll, dim=-1), dim=0)[None]
    return mean_kde_ll

  
class DynamicBufferModule(ABC, torch.nn.Module):
    """Torch module that allows loading variables from the state dict even in the case of shape mismatch."""
    
    def get_tensor_attribute(self, attribute_name: str) -> Tensor:
        """Get attribute of the tensor given the name.
        Args:
            attribute_name (str): Name of the tensor
        Raises:
            ValueError: `attribute_name` is not a torch Tensor
        Returns:
            Tensor: Tensor attribute
        """
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, Tensor):
            return attribute
        raise ValueError(f"Attribute with name '{attribute_name}' is not a torch Tensor")

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args):
        """Resizes the local buffers to match those stored in the state dict.
        Overrides method from parent class.
        Args:
          state_dict (dict): State dictionary containing weights
          prefix (str): Prefix of the weight file.
          *args:
        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers.keys():
            for key in state_dict.keys():
                if key.startswith(prefix) and key[len(prefix) :].split(".")[0] == param:
                    if not local_buffers[param].shape == state_dict[key].shape:
                        attribute = self.get_tensor_attribute(param)
                        attribute.resize_(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args)
        

class GaussianKDE(DynamicBufferModule):
    """Gaussian Kernel Density Estimation.
    Args:
        dataset (Optional[Tensor], optional): Dataset on which to fit the KDE model. Defaults to None.
    """

    def __init__(self, dataset: Optional[Tensor] = None):
        super().__init__()

        self.register_buffer("bw_transform", Tensor())
        self.register_buffer("dataset", Tensor())
        self.register_buffer("norm", Tensor())
        
        if dataset is not None:
            self.fit(dataset)
        
        
    def forward(self, features: Tensor) -> Tensor:
        """Get the KDE estimates from the feature map.
        Args:
          features (Tensor): Feature map extracted from the CNN
        Returns: KDE Estimates
        """
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0]).to(features.device)
        for i in range(features.shape[0]):
            embedding = ((self.dataset - features[i]) ** 2).sum(dim=1)
            embedding = self.log_norm - (embedding / 2)
            estimate[i] = torch.mean(embedding)
        return estimate


    def fit(self, dataset: Tensor) -> None:
        """Fit a KDE model to the input dataset.
        Args:
          dataset (Tensor): Input dataset.
        Returns:
            None
        """        
        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4))

        cov_mat = self.cov(dataset.T)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor**2
        
        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)
        
        #
        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm
        self.log_norm = torch.log(self.norm)
        return


    @staticmethod
    def cov(tensor: Tensor) -> Tensor:
        """Calculate the unbiased covariance matrix.
        Args:
            tensor (Tensor): Input tensor from which covariance matrix is computed.
        Returns:
            Output covariance matrix.
        """
        mean = torch.mean(tensor, dim=1, keepdim=True)
        cov = torch.matmul(tensor - mean, (tensor - mean).T) / (tensor.size(1) - 1)
        return cov
    


def denomarlize(*data):
    out = []
    for x in data:
        x = x * dataset.std + dataset.mean
        out.append(x)
    return out


def get_prediction(data, algo, sample_num, num_seeds=1, concat_hist=True):
    traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    X = traj[:t_his]

    if algo == 'dlow':
        X = X.repeat((1, num_seeds, 1))
        Z_g = models[algo].sample(X)
        X = X.repeat_interleave(nk, dim=1)
        Y = models['vae'].decode(X, Z_g)
    elif algo == 'vae':
        X = X.repeat((1, sample_num * num_seeds, 1))
        Y = models[algo].sample_prior(X)

    if concat_hist:
        Y = torch.cat((X, Y), dim=0)
    Y = Y.permute(1, 0, 2).contiguous().cpu().numpy()
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y



def compute_stats():
    data_gen = dataset.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = args.num_seeds
    
    kde_list = []
    for i, data in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data) # (1, 60, 42)
        gt = np.reshape(gt, newshape=(gt.shape[0], gt.shape[1], -1, 3))
        for algo in algos:
            if algo == 'vae':
                continue

            pred_thousand = []
            for j in range(20):
                pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False) # (1, 50, 60, 42)
                pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1], pred.shape[2], -1, 3))
                pred_thousand.append(pred)
            pred_thousand = np.concatenate(pred_thousand, axis=1)
            pred = pred_thousand

            '''
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False) # (1, 50, 60, 42)
            pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1], pred.shape[2], -1, 3))
            '''
            kde_list.append(kde(y=gt, y_pred=pred))
    kde_ll = torch.cat(kde_list, dim=0)
    kde_ll = torch.mean(kde_ll, dim=0)
    kde_ll_np = kde_ll.to('cpu').numpy()
    print(kde_ll_np)
        
    
    
def visualize():

    def post_process(pred, data):
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
        if cfg.normalize_data:
            pred = denomarlize(pred)
        pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
        pred[..., :1, :] = 0
        return pred

    def pose_generator():

        while True:
            data = dataset.sample()

            # gt
            gt = data[0].copy()
            gt[:, :1, :] = 0
            poses = {'context': gt, 'gt': gt}
            # vae
            for algo in vis_algos:
                pred = get_prediction(data, algo, nk)[0]
                pred = post_process(pred, data)
                for i in range(pred.shape[0]):
                    poses[f'{algo}_{i}'] = pred[i]

            yield poses

    pose_gen = pose_generator()
    render_animation(dataset.skeleton, pose_gen, vis_algos, cfg.t_his, ncol=12, output='out/video.mp4')


def get_gt(data):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]


"""metrics"""

def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_ade(pred, gt, *args):
    diff = pred - gt
    temp = np.linalg.norm(diff, axis=2)
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def get_multimodal_gt():
    all_data = []
    data_gen = dataset.iter_generator(step=cfg.t_his)
    for data in data_gen:
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
    return traj_gt_arr


if __name__ == '__main__':
    all_algos = ['dlow', 'vae']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='humaneva_nsamp50')
    parser.add_argument('--mode', default='stats')
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=None)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        iter_algo = 'iter_%s' % algo
        num_algo = 'num_%s_epoch' % algo
        setattr(args, iter_algo, getattr(cfg, num_algo))
        algos.append(algo)
    vis_algos = algos.copy()

    if args.action != 'all':
        args.action = set(args.action.split(','))

    """parameter"""
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred

    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls(args.data, t_his, t_pred, actions=args.action, use_vel=cfg.use_vel)
    # traj_gt_arr = get_multimodal_gt()

    """models"""
    model_generator = {
        'vae': get_vae_model,
        'dlow': get_dlow_model,
    }
    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg, dataset.traj_dim)
        model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, f'iter_{algo}')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = pickle.load(open(model_path, "rb"))
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

    if cfg.normalize_data:
        dataset.normalize_data(model_cp['meta']['mean'], model_cp['meta']['std'])

    if args.mode == 'vis':
        visualize()
    elif args.mode == 'stats':
        compute_stats()