import torch

from pathlib import Path
from cleanfid import fid
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def store_FID_baseline(baseline_stats_name: str, image_dir: str|Path, device):
    '''
    Stores a baseline in the cleanFID cache. Calculates the mean and variance of the 
    dataset distribution, and stores it in cache under the name [baseline_stats_name]

    @param baseline_stats_name: name of the baseline
    @param image_dir: directory of the images to use to calculate the baseline
    @param device: torch device
    '''

    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {image_dir}")
    
    fid.make_custom_stats(name=baseline_stats_name, fdir=str(image_dir), device=device, num_workers=0)


def compute_FID(baseline_stats_name: str, dir: str, device: torch.device, resolution: int = 256) -> float:
    '''
    Calculate the FID score between images in dir, and precomputed dataset statstics. Uses a pretrained
    network to extract low frequency information from the images, and compares them.

    Can be used to calculate the rFID (between real and constructed images (stored in dir)), 
    and the gFID (between real and generated images (stored in dir)).

    A baseline should be stored under the the name [baseline_stats_name]

    @param real_stats_name: 
    @param dir: directory where 
    @param device: torch device
    @param resolution: resolution used to compute stats 
    '''

    return fid.compute_fid(
        fdir1=dir,
        dataset_name=baseline_stats_name,
        dataset_res=resolution,
        dataset_split="custom",
        device=device,
        num_workers=0
    )


def compute_PSNR(reconstructed: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    '''
    Calculates the Peak Signal-to-Noise Ratio for measuring reconstruction quality.  

    @param reconstructed: the reconstructed images as a Tensor (B,C,H,W)
    @param target: ground truth images as a Tensor (B,C,H,W)
    @param data_range: expected range [0, 1]
    '''

    metric = PeakSignalNoiseRatio(data_range=data_range)
    return metric(reconstructed, target).item()
    

def compute_SSIM(reconstructed: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    '''
    Calculates the Structural Similarity Index Measure for measuring reconstruction quality.  

    @param reconstructed: the reconstructed images as a Tensor (B,C,H,W)
    @param target: ground truth images as a Tensor (B,C,H,W)
    @param data_range: expected range [0, 1]
    '''
    
    metric = StructuralSimilarityIndexMeasure(data_range=data_range)
    return metric(reconstructed, target).item()