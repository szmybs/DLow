import numpy as np
import argparse
import os
import sys
import pickle
import csv
import re
from scipy.spatial.distance import pdist

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m import DatasetH36M
from motion_pred.utils.dataset_humaneva import DatasetHumanEva
from motion_pred.utils.visualization import render_animation
from models.motion_pred import *
from scipy.spatial.distance import pdist, squareform
from FID.fid import fid
from FID.fid_classifier import classifier_fid_factory, classifier_fid_humaneva_factory


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


def compute_stats():
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}

    data_gen = dataset.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = args.num_seeds
    for i, data in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data)
        gt_multi = traj_gt_arr[i]
        for algo in algos:
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            pred = pred[:, [0, 2, 4, 6, 8]]
            for stats in stats_names:
                val = 0
                for pred_i in pred:
                    val += stats_func[stats](pred_i, gt, gt_multi) / num_seeds
                stats_meter[stats][algo].update(val)
        print('-' * 80)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join([f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats)

    logger.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger.info(str_stats)
    logger.info('=' * 80)

    with open('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + algos)
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
        

def FID_test(classifier):
    data_gen = dataset.iter_generator(step=cfg.t_his, afg=True)
    num_samples = 0
    num_seeds = args.num_seeds
    
    pred_act_list, gt_act_list = [], []
    for i, (data, action) in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data)
        
        gt = np.repeat(gt, 50, axis=0)
        gt = np.swapaxes(gt, 1, 2)
        gt = torch.tensor(gt, device=device)        
        
        for algo in algos:
            if algo == 'vae':
                continue
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            pred = np.swapaxes(pred, -2, -1)
            
            # pred = pred.reshape(50, 48, 100)
            pred = pred.reshape(10, 48, 100)
            pred = torch.tensor(pred, device=device, dtype=torch.float32)
            pred = pred[[0, 2, 4, 6, 8], ...]

            # # pred = pred.reshape(50, 42, 60)
            # pred = pred.reshape(10, 42, 60)
            # pred = torch.tensor(pred, device=device, dtype=torch.float32)
            # pred = pred[[0, 2, 4, 6, 8], ...]

            pred_activations = classifier.get_fid_features(motion_sequence=pred).cpu().data.numpy()
            gt_activations   = classifier.get_fid_features(motion_sequence=gt).cpu().data.numpy()
        
            pred_act_list.append(pred_activations)
            gt_act_list.append(gt_activations)
        
    results_fid = fid(np.concatenate(pred_act_list, 0), np.concatenate(gt_act_list, 0))
    print(results_fid)


def CMD_test():
    idx_to_class = ['directions', 'discussion', 'eating', 'greeting', 'phoning', \
                    'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',  \
                    'photo', 'waiting', 'walking', 'walkdog', 'walktogether']
    # idx_to_class = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', \
    #                 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',  \
    #                 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
    mean_motion_per_class = [0.004528946212615328, 0.005068199383505345, 0.003978791804673771,  0.005921345536787865,   0.003595039379111546, 
                            0.004192961478268034, 0.005664689143238568, 0.0024945400286369122, 0.003543066357658834,   0.0035990843311130487, 
                            0.004356865838457266, 0.004219841185066826, 0.007528046315984569,  0.00007054820734533077, 0.006751761745020258]  

    def CMD(val_per_frame, val_ref):
        T = len(val_per_frame) + 1
        return np.sum([(T - t) * np.abs(val_per_frame[t-1] - val_ref) for t in range(1, T)])

    def CMD_helper(pred):
        pred_flat = pred   # shape: [batch, num_s, t_pred, joint, 3]
        # M = (torch.linalg.norm(pred_flat[:,:,1:] - pred_flat[:,:,:-1], axis=-1)).mean(axis=1).mean(axis=-1)    
        M = np.linalg.norm(pred_flat[:,:,1:] - pred_flat[:,:,:-1], axis=-1).mean(axis=1).mean(axis=-1) 
        return M

    def CMD_pose(data, label):
        ret = 0
        # CMD weighted by class
        for i, (name, class_val_ref) in enumerate(zip(idx_to_class, mean_motion_per_class)):
            mask = label == name
            if mask.sum() == 0:
                continue
            motion_data_mean = data[mask].mean(axis=0)
            ret += CMD(motion_data_mean, class_val_ref) * (mask.sum() / label.shape[0])
        return ret

    data_gen = dataset.iter_generator(step=cfg.t_his, afg=True)
    num_samples = 0
    num_seeds = 1

    M_list, label_list = [], []
    for data, action in data_gen:
        for algo in algos:
            if algo == 'vae':
                continue
            action = str.lower(re.sub(r'[0-9]+', '', action))
            action = re.sub(" ", "", action)
            
            num_samples += 1
            gt = get_gt(data)
            
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            pred = pred.reshape(1, 10, 100, 16, 3)
            pred = pred[:, (2, 4, 6, 8, 0)]
            
            M = CMD_helper(pred)
            M_list.append(M)
            label_list.append(action)

    M_all = np.concatenate(M_list, 0)
    label_all = np.array(label_list)
    
    cmd_score = CMD_pose(M_all, label_all) 
    print(cmd_score)
    return


def CMD_test_eva():
    idx_to_class = ['Box', 'Gestures', 'Jog', 'ThrowCatch', 'Walking']
    mean_motion_per_class = [0.010139551, 0.0021507503, 0.010850595,  0.004398426,   0.006771291]  

    def CMD(val_per_frame, val_ref):
        T = len(val_per_frame) + 1
        return np.sum([(T - t) * np.abs(val_per_frame[t-1] - val_ref) for t in range(1, T)])

    def CMD_helper(pred):
        pred_flat = pred   # shape: [batch, num_s, t_pred, joint, 3] 
        M = np.linalg.norm(pred_flat[:,:,1:] - pred_flat[:,:,:-1], axis=-1).mean(axis=1).mean(axis=-1) 
        return M

    def CMD_pose(data, label):
        ret = 0
        # CMD weighted by class
        for i, (name, class_val_ref) in enumerate(zip(idx_to_class, mean_motion_per_class)):
            mask = label == name
            if mask.sum() == 0:
                continue
            motion_data_mean = data[mask].mean(axis=0)
            ret += CMD(motion_data_mean, class_val_ref) * (mask.sum() / label.shape[0])
        return ret

    data_gen = dataset.iter_generator(step=cfg.t_his, afg=True)
    num_samples = 0
    num_seeds = 1

    M_list, label_list = [], []
    for data, action in data_gen:
        for algo in algos:
            if algo == 'vae':
                continue
            
            num_samples += 1
            gt = get_gt(data)
            
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            pred = pred.reshape(1, 50, 60, 14, 3)
            # pred = pred.reshape(1, 10, 60, 14, 3)
            # pred = pred[:, (2, 4, 6, 8, 0)]
            
            M = CMD_helper(pred)
            M_list.append(M)
            label_list.append(action)

    M_all = np.concatenate(M_list, 0)
    label_all = np.array(label_list)
    
    cmd_score = CMD_pose(M_all, label_all) 
    print(cmd_score)

    return


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
    parser.add_argument('--cfg', default='h36m_nsamp10')
    # parser.add_argument('--cfg', default='humaneva_nsamp50')
    parser.add_argument('--mode', default='FID')
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=-1)
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=None)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
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
    traj_gt_arr = get_multimodal_gt()

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
        
    elif args.mode == 'FID':
        classifier = classifier_fid_factory(device) if cfg.dataset == 'h36m' else classifier_fid_humaneva_factory(device)
        FID_test(classifier)

    elif args.mode == 'CMD':
        with torch.no_grad():
            if cfg.dataset == 'h36m':
                CMD_test()
            else:
                CMD_test_eva() 
