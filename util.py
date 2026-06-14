# modified from https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import builtins
import datetime
import glob
import math
import os
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.distributed as dist
import yaml
from torch import inf
from torch.utils.data import DataLoader, Dataset

from dl_utils import NUCLEUS_LABEL_KEY, NUCL_TABLE, LM_DF





def adjust_learning_rate(optimizer, epoch, args, start_epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs

    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) /
             (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('GPU max mem (MB): {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def print_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(rank, world_size, args):

    if args.dist_on_itp:

        os.environ['MASTER_PORT'] = '12355'
        os.environ['MASTER_ADDR'] = args.dist_url
        args.rank = rank # Process ID
        args.world_size = world_size  # Number of processes, if nproc of torchrun is > 1
        args.gpu = rank # Number of processes, if nproc of torchrun is > 1

    else:
        print('Not using distributed mode')
        print_for_distributed(is_master=True)  # hack
        args.dist_on_itp = False
        return args

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.cuda.set_device(args.gpu)
    # initialize the process group
    dist.init_process_group(args.dist_backend, rank=rank, world_size=world_size)
    print_for_distributed(args.rank == 0)

    return args

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, wb_run):
    output_dir = args.output_dir
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args
            }
            if wb_run is not None:
                to_save['wb_run_id'] = wb_run.id
                to_save['wb_run_url'] = wb_run.url

            torch.save(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def loadcheckpoints(train_run_path: str):
    return sorted(glob.glob(os.path.join(train_run_path, '*.pth')), key=os.path.getmtime)


def getcheckpointinfoandargs(pnt: str):
    print("Predicting features for", pnt)
    checkpoint_info = torch.load(pnt, map_location='cpu', weights_only=False)
    cpt_args = checkpoint_info['args']
    return checkpoint_info, cpt_args


def get_model_dataset_cpt_args(get_dataset, cpt_args, batch_size, num_workers, pin_memory, drop_last=False):

    model, dataset = setup(get_dataset, cpt_args)

    data_loader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    return model, data_loader, dataset


def get_dataset(LightMicroscope: bool = False, nucl_vol_diameter=11, num_patch_per_nucl=1331, mask_ratio=0.8,
                num_sin_cos_pos_emb=78, encoder_embed_dim=80, texture_patch_dim=4, test=False, **kwargs) -> Dataset:

    from platy_nuclei_texture.model_dataset_utils.nuclei_loader import LMTextureNucleiDataset, EMTextureNucleiDataset
    if LightMicroscope is True:
        print("We are dealing with LM data")
        dset = LMTextureNucleiDataset(nucl_vol_diameter=nucl_vol_diameter,
                                      num_patch_per_nucl=num_patch_per_nucl,
                                      mask_ratio=mask_ratio,
                                      num_sin_cos_pos_emb=num_sin_cos_pos_emb,
                                      encoder_embed_dim=encoder_embed_dim,
                                      texture_patch_dim=texture_patch_dim,
                                      test=test, **kwargs)

    else:
        print("We are dealing with EM data")
        dset = EMTextureNucleiDataset(nucl_vol_diameter=nucl_vol_diameter, num_patch_per_nucl=num_patch_per_nucl,
                                      mask_ratio=mask_ratio, num_sin_cos_pos_emb=num_sin_cos_pos_emb,
                                      encoder_embed_dim=encoder_embed_dim, test=test, texture_patch_dim=32, **kwargs)
    return dset


def get_model(LightMicroscope: bool = False, mask_ratio=0.8, embed_dim=80, encoder_embed_dim=80, encoder_depth=12,
              encoder_num_heads=4, decoder_embed_dim=80, decoder_depth=8, decoder_num_heads=4, mlp_ratio=4,
              texture_patch_dim=32, encoder_type='resnet', resnet_layers=[2, 2], resnet_block='BasicBlock',
              resnet_inplanes=64, resnet_after_pool_size=1, recontr_downsample=4, downsample_mode='area',
              loss_mask_zero=False, mae_encoder=False, final_activation="None", mask_only=False, norm_pix_loss=False,
              pretrained=None, memory_efficiency=False, num_patch_per_nucl=200, **kwargs) -> torch.nn.Module:

    from platy_nuclei_texture.model_dataset_utils.nuclei_models import LMAutoencoderViT, RawMaskedAutoencoderViT
    
    if LightMicroscope is True:

        model = LMAutoencoderViT(texture_patch_dim=texture_patch_dim, embed_dim=embed_dim, encoder_embed_dim=encoder_embed_dim,
                                        encoder_type=encoder_type, resnet_layers=resnet_layers,
                                        resnet_block=resnet_block, resnet_inplanes=resnet_inplanes,
                                        resnet_after_pool_size=resnet_after_pool_size,
                                        mask_ratio=mask_ratio, encoder_depth=encoder_depth,
                                        encoder_num_heads=encoder_num_heads, decoder_embed_dim=decoder_embed_dim,
                                        decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                                        mlp_ratio=mlp_ratio, mae_encoder=mae_encoder, final_act=final_activation, masked_loss=mask_only,
                                        loss_mask_zero=loss_mask_zero, norm_pix_loss=norm_pix_loss, pretrained=pretrained, memory_efficiency=memory_efficiency,
                                 num_patch_per_nucl=num_patch_per_nucl, **kwargs)

    else:

        model = RawMaskedAutoencoderViT(embed_dim=embed_dim, encoder_embed_dim=encoder_embed_dim,
                                        encoder_type=encoder_type, resnet_layers=resnet_layers,
                                        resnet_block=resnet_block, resnet_inplanes=resnet_inplanes,
                                        resnet_after_pool_size=resnet_after_pool_size,
                                        recontr_downsample=recontr_downsample, downsample_mode=downsample_mode,
                                        mask_ratio=mask_ratio, encoder_depth=encoder_depth,
                                        encoder_num_heads=encoder_num_heads, decoder_embed_dim=decoder_embed_dim,
                                        decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                                        mlp_ratio=mlp_ratio, mae_encoder=mae_encoder, final_act=final_activation, masked_loss=mask_only,
                                        loss_mask_zero=loss_mask_zero, norm_pix_loss=norm_pix_loss, pretrained=pretrained, memory_efficiency=memory_efficiency,
                                        num_patch_per_nucl=num_patch_per_nucl, **kwargs)

    return model


def setup(get_dataset, args: argparse.Namespace):

    model_dataset_args = vars(args)

    dataset = get_dataset(**model_dataset_args)

    model = get_model(**model_dataset_args)

    return model, dataset

@torch.inference_mode()
def get_output_dict(dataset, model, save_dir, df_index=7685, device="cpu"):
    model.to(device)
    model.eval()

    input_ = [dataset.__getitem__(df_index)]

    s = os.path.join(save_dir, f"random_masking_{df_index}.npy")
    if not os.path.exists(s):
        os.makedirs(os.path.dirname(s), exist_ok=True)
        np.save(s, input_[0][3])
    enc_ids = np.load(s)

    if not enc_ids.shape == input_[0][3].shape:
        enc_ids = input_[0][3]
    # NOTE: enc_ids does not correspond to map_ids anymore.
    x = input_[0]
    input_ = [x[0], x[1], x[2], enc_ids, np.array(x[4]), x[5]]

    loss, output_dict = model.forward_wo_dataloader(input_, device)
    
    print(f"The loss is {loss}")
    return output_dict

def save2DFcolumn(
    sorted_results: list,
    sorted_nucl_labels: np.ndarray,
    dataframe: pd.DataFrame,
    column_name: str = "Embedding"
) -> pd.DataFrame:
    """
    Adds a new column to the dataframe with values from sorted_results,
    mapped according to sorted_nucl_labels.

    Parameters:
    - sorted_results: List of values to be added as the new column.
    - sorted_nucl_labels: 1D or 2D numpy array of nucleus labels.
    - dataframe: The DataFrame to which the new column will be added.
    - column_name: The name of the new column (default is "Embedding").

    Returns:
    - Updated DataFrame with the new column.
    """

    if not isinstance(sorted_nucl_labels, np.ndarray):
        sorted_nucl_labels = np.array(sorted_nucl_labels)
    # Flatten sorted_nucl_labels if it has only one column (2D array with shape [n, 1])
    if sorted_nucl_labels.ndim == 2:
        if sorted_nucl_labels.shape[0] == 1 or sorted_nucl_labels.shape[1] == 1:
            if len(sorted_nucl_labels) == sorted_nucl_labels.size:
                sorted_nucl_labels = sorted_nucl_labels.flatten()
            else:
                raise  ValueError("Sorted nucleus labels and sorted results do not match in length")
        else:
            raise NotImplementedError("Handling for multi-column sorted_nucl_labels is not implemented.")

    if isinstance(sorted_results, np.ndarray):

        if sorted_results.ndim > 2:
            raise NotImplementedError("Handling for multi-column sorted_nucl_labels is not implemented.")
        else:
            sorted_results = sorted_results.tolist()

    # Create a dictionary for fast lookup of results by label
    label_to_result = dict(zip(sorted_nucl_labels, sorted_results))

    # Map each NUCLEUS_LABEL_KEY in the dataframe to its corresponding result, or NaN if not found
    dataframe[column_name] = dataframe.label_id.map(label_to_result).fillna(np.nan)

    return dataframe


def savedataframe(dataframe, save_dir,  **kwargs):
    # Remove unnamed columns
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
    dataframe.drop(columns=dataframe.columns[dataframe.columns.duplicated()], inplace=True)
    dataframe.drop_duplicates(subset=NUCLEUS_LABEL_KEY, inplace = True)

    dataframe.reset_index(inplace=True, drop=True)

    if not "modality" in dataframe.columns:
        if "LM" in save_dir:
            dataframe["modality"] = "LM"
        elif "EM" in save_dir:
            dataframe["modality"] = "EM"

    dataframe.to_json(get_savedf_path(save_dir, **kwargs))

def get_savedf_path(save_dir: str, typie=''):

    if typie != "":
        return os.path.join(save_dir, f"dataframe_{typie}.json")
    else:
        return os.path.join(save_dir, f"dataframe.json")


def readdataframe(path: str, name='') -> pd.DataFrame:

    if "json" in path and "dataframe" in path and os.path.exists(path):
        data_df = pd.read_json(path)
    else:
        assert name is not None, ("If path is save_dir then please do not set parameter 'name' as None")

        if os.path.exists(path):
            path = get_savedf_path(path, typie=name)

        if not os.path.exists(path):
            print(f"Provided path does not exist: {path}")
            return None

        data_df = pd.read_json(path)

    if not "modality" in data_df.columns:

        if "LM" in path:
            data_df["modality"] = "LM"
        elif "EM" in path:
            data_df["modality"] = "EM"

    return data_df


def getvalidationdataloader(dataset, batch_size, num_workers, pin_memory, drop_last):

    # TODO: Place it here for now to avoid import conflicts
    from platy_nuclei_texture.model_dataset_utils.helperfunctions import addcell_type_col, cell_type2class_column


    dataset.data_df = addcell_type_col(dataset.data_df)

    # Update the dataset's DataFrame with cell type classifications
    dataset.data_df = cell_type2class_column(dataset.data_df)

    labels = dataset.data_df.dropna(subset=["cell_type", NUCLEUS_LABEL_KEY]).label_id.to_numpy().flatten()

    subset_dataset = dataset.subset_dataset(labels, nucl=True)

    data_loader = DataLoader(subset_dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    return data_loader, subset_dataset.data_df


class CustomArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        # Parse known arguments
        args = super().parse_args(*args, **kwargs)

        # Identify specified and default arguments
        specified = {}
        defaults = {}

        for action in self._actions:
            arg = action.dest
            if arg is not None and hasattr(args, arg):  # Skip actions without a dest attribute
                value = getattr(args, arg)
                if value != action.default:
                    specified[arg] = value
                else:
                    defaults[arg] = value

        return args, specified, defaults


def update_args_from_yaml(args, config_file):
    if config_file is not None:
        with open(config_file, "r") as file:
            yaml_args = yaml.safe_load(file)
            for key, value in yaml_args.items():
                setattr(args, key, value)
    return args


def clean_config(config: dict):
    """
    Clean the configuration dictionary by:
    - Removing keys with None values.
    - Ensuring values are of type float, int, str, or None.
    - (Optional) Sorting keys alphabetically.

    Args:
        config (dict): Configuration dictionary to clean.

    Returns:
        dict: Cleaned configuration dictionary.
    """
    # Allowed value types
    allowed_types = (float, int, str, type(None))

    # Remove keys with invalid or None values
    cleaned_config = {
        k: v for k, v in config.items()
        if isinstance(v, allowed_types)
    }

    return cleaned_config


def cleanup():
    dist.destroy_process_group()

def log_memory_usage():
    process = psutil.Process()  # Get current process
    memory_info = process.memory_info()  # Get memory details
    print(f"RSS (Resident Set Size): {memory_info.rss / 1e6:.2f} MB")
    print(f"VMS (Virtual Memory Size): {memory_info.vms / 1e6:.2f} MB")


def save_img(label, masked=None, img=None, mask=None, savename="", save_dir=os.getcwd()):

    filename = f"{label}_"
    if masked is not None:
        path = os.path.join(save_dir, filename + "masked_" + savename + ".npy")
        np.save(path, masked)

    elif img is not None:
        path = os.path.join(save_dir, filename + "img_" + savename + ".npy")
        np.save(path, img)

    elif mask is not None:
        path = os.path.join(save_dir, filename + "mask_" + savename + ".npy")
        np.save(path, mask)


def mkdir(root_dir, dirname="patches"):

    dir = os.path.join(root_dir, dirname)

    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir


def mkresultsdir(training_dir):
    return mkdir(training_dir, dirname="results")


def mkepochdir(training_dir, epoch):

    save_dir = mkresultsdir(training_dir)

    epochdir = os.path.join(save_dir, f"epoch_{epoch}")

    if not os.path.exists(epochdir):
        os.mkdir(epochdir)

    return epochdir


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def patchify(vol, patch_size):
    """
    Extract patches from the input volume, agnostic to 2D or 3D inputs.

    Parameters:
    vol: numpy.ndarray of shape (Y, X) for 2D or (Z, Y, X) for 3D
        Each spatial dimension must be divisible by patch_size.

    Returns:
    numpy.ndarray of shape (L, *([patch_size] * ndim))
        2D: (L, p, p)      with L = (Y/p) * (X/p)
        3D: (L, p, p, p)   with L = (Z/p) * (Y/p) * (X/p)
    """
    p = patch_size
    ndim = vol.ndim

    assert all(s % p == 0 for s in vol.shape), \
        f"Input dimensions {vol.shape} must be divisible by the patch size {p}."

    grid = tuple(s // p for s in vol.shape)  # patches per spatial axis

    # Reshape so each spatial axis splits into (grid_i, p):
    #   2D: (h, p, w, p)      3D: (d, p, h, p, w, p)
    split_shape = tuple(x for g in grid for x in (g, p))
    x = vol.reshape(split_shape)

    # Group all grid axes first, then all patch axes:
    #   2D: 'hpwq->hwpq'      3D: 'dfhpwq->dhwfpq'
    src = ''.join(chr(ord('a') + i) for i in range(2 * ndim))
    grid_axes = src[0::2]   # even positions = grid indices
    patch_axes = src[1::2]  # odd positions = within-patch indices
    x = np.einsum(f'{src}->{grid_axes + patch_axes}', x)

    return x.reshape(int(np.prod(grid)), *([p] * ndim))


def merge_with_nucl_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a nucleus DataFrame with the global nucleus table (NUCL_TABLE for EM,
    LM_DF for LM) to attach bounding-box and anchor columns.

    The DataFrame must contain a 'modality' column ('EM' or 'LM').
    """
    assert "modality" in df.columns, \
        f"Please specify modality in dataframe: {list(df.columns)}"

    if "EM" in df["modality"].unique().flatten():
        left_df = NUCL_TABLE
    else:
        left_df = LM_DF

    left_df = left_df.loc[:, [
        "label_id",
        "anchor_z", "anchor_x", "anchor_y",
        "bb_min_z", "bb_max_z", "bb_min_y", "bb_max_y", "bb_min_x", "bb_max_x",
    ]]

    assert df.label_id.isin(left_df.label_id).sum() == df.label_id.size, (
        f"Dataframe does not have nuclei labels as label_id: "
        f"{df.label_id.isin(left_df.label_id).sum()}/{df.label_id.size}"
    )

    df = df.merge(left_df, on="label_id", how="left", suffixes=(None, "_nucl"))

    cols2drop = [col for col in df.columns if col.endswith("_nucl")]
    if cols2drop:
        df.drop(columns=cols2drop, inplace=True)

    return df
