import numpy as np
import numpy_indexed as npi
import json

import torch


def sliding_blocks_point_indices(pts, block_size, overlap_ratio):
    """
    Slice 2D or 3D points into overlapping blocks.

    Parameters
    ----------
    pts : (N, 2) or (N, 3) array
        Input points, either 2D or 3D.
    block_size : sequence of length 2 or 3
        Size of each block in each dimension.
    overlap_ratio : float in [0,1)
        Fractional overlap between adjacent blocks.

    Returns
    -------
    origins : (M, 2) or (M, 3) array
        Array of each block’s lower-corner coordinates.
    groups : list of ndarrays
        For each block, an array of point indices that fall inside it.
    """
    pts = np.asarray(pts, float)
    ndim = pts.shape[1]
    if ndim not in (2, 3):
        raise ValueError(
            "Only 2D or 3D points are supported (shape (N,2) or (N,3)).")

    bs = np.asarray(block_size, float)
    if bs.size != ndim:
        raise ValueError(f"block_size must have length {ndim}.")
    stride = bs * (1.0 - overlap_ratio)

    p_min = pts.min(axis=0)
    p_max = pts.max(axis=0)
    # Number of stride steps needed so the last block still reaches p_max.
    # Clamped at 0: when an axis spans less than (block - stride), the raw
    # value goes negative, ``dims`` drops to 0, every point collapses into
    # one bogus block (index -1) and the voxel filter downstream then keeps
    # only the corner nearest the minimum. A single block per axis always
    # covers such a thin extent, e.g. a flat tile whose Z range is under
    # 10 % of the 51.2 m Z block (farmland, water, polders).
    steps = np.floor((p_max - p_min - bs) / stride).astype(int) + 1
    steps = np.maximum(steps, 0)
    dims = steps + 1  # number of blocks along each axis

    # Compute each point’s “primary” block index along each dimension
    rel = (pts - p_min) / stride
    idx0 = np.floor(rel).astype(int)  # shape (N, ndim)

    # Prepare indices for the current block and the previous block (clipped)
    ixs = np.clip(
        np.stack([idx0[:, 0], idx0[:, 0] - 1], axis=1), 0, dims[0] - 1)
    iys = np.clip(
        np.stack([idx0[:, 1], idx0[:, 1] - 1], axis=1), 0, dims[1] - 1)
    N = pts.shape[0]

    if ndim == 3:
        izs = np.clip(
            np.stack([idx0[:, 2], idx0[:, 2] - 1], axis=1), 0, dims[2] - 1)
        combos = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        ix = np.stack([ixs[np.arange(N), a] for a, _, _ in combos], axis=1)
        iy = np.stack([iys[np.arange(N), b] for _, b, _ in combos], axis=1)
        iz = np.stack([izs[np.arange(N), c] for _, _, c in combos], axis=1)

        cond_x = (pts[:, 0, None] >= p_min[0] + ix * stride[0]
                  ) & (pts[:, 0, None] < p_min[0] + ix * stride[0] + bs[0])
        cond_y = (pts[:, 1, None] >= p_min[1] + iy * stride[1]
                  ) & (pts[:, 1, None] < p_min[1] + iy * stride[1] + bs[1])
        cond_z = (pts[:, 2, None] >= p_min[2] + iz * stride[2]
                  ) & (pts[:, 2, None] < p_min[2] + iz * stride[2] + bs[2])
        mask = cond_x & cond_y & cond_z

        block_ids = (ix * (dims[1] * dims[2]) + iy *
                     dims[2] + iz).ravel()[mask.ravel()]
    else:
        # 2D case: only 4 combinations of (ix,iy)
        combos = [(a, b) for a in (0, 1) for b in (0, 1)]
        ix = np.stack([ixs[np.arange(N), a] for a, _ in combos], axis=1)
        iy = np.stack([iys[np.arange(N), b] for _, b in combos], axis=1)

        cond_x = (pts[:, 0, None] >= p_min[0] + ix * stride[0]
                  ) & (pts[:, 0, None] < p_min[0] + ix * stride[0] + bs[0])
        cond_y = (pts[:, 1, None] >= p_min[1] + iy * stride[1]
                  ) & (pts[:, 1, None] < p_min[1] + iy * stride[1] + bs[1])
        mask = cond_x & cond_y

        block_ids = (ix * dims[1] + iy).ravel()[mask.ravel()]

    point_ids = np.repeat(np.arange(N), mask.shape[1])[mask.ravel()]
    unique_bids, groups = npi.group_by(block_ids, point_ids)

    # Decode block origins
    if ndim == 3:
        ix_u = unique_bids // (dims[1] * dims[2])
        rem = unique_bids % (dims[1] * dims[2])
        iy_u = rem // dims[2]
        iz_u = rem % dims[2]
        origins = np.vstack([
            p_min[0] + ix_u * stride[0],
            p_min[1] + iy_u * stride[1],
            p_min[2] + iz_u * stride[2],
        ]).T
    else:
        ix_u = unique_bids // dims[1]
        iy_u = unique_bids % dims[1]
        origins = np.vstack([
            p_min[0] + ix_u * stride[0],
            p_min[1] + iy_u * stride[1],
        ]).T

    return origins, [np.array(g, int) for g in groups]


def resolve_device(device) -> str:
    """Validate the requested compute device and return its torch name.

    Accepts ``"cuda"``, ``"mps"``, ``"cpu"`` (and the v1.0 booleans:
    ``True`` -> cuda, ``False`` -> cpu). Raises a RuntimeError with an
    actionable message instead of letting torch fail deep inside the
    model: v1.0.2 called ``model.cuda()`` for any "GPU" choice, which on
    Apple Silicon (where the probe reports MPS) died with a bare
    ``AssertionError: Torch not compiled with CUDA enabled``.
    """
    if device is True:
        device = "cuda"
    if device is False or device is None:
        device = "cpu"
    device = str(device).lower()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this torch build has no working "
                f"CUDA device (torch {torch.__version__}, CUDA build: "
                f"{torch.version.cuda}). Choose CPU, or reinstall the "
                "dependencies from the Setup panel."
            )
        return device
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "Apple MPS was requested but is not available in this "
                "torch build. Choose CPU."
            )
        return device
    if device == "cpu":
        return device
    raise RuntimeError(
        f"Unknown compute device '{device}'. Use 'cuda', 'mps' or 'cpu'."
    )


def load_segformer(config_file, model_path, device):
    """Build the 3D SegFormer from its JSON and load the weights onto ``device``.

    Returns ``(model, configs)``; ``configs`` carries the block geometry
    that ``predict_blocks`` needs. Loading once per run and predicting per
    tile is what the backends do; ``filterPoints`` below keeps the v1.0
    one-call behaviour.
    """
    device = resolve_device(device)
    # Load the block geometry and the network hyper-parameters from the
    # model's JSON (e.g. urbanfiltering_als_esegformer3D_112_30cm...).
    try:
        with open(config_file) as json_file:
            configs = json.load(json_file)
    except Exception as exc:
        # Callers treat a None result as "no predictions" and skip the
        # file; a missing or broken config must stop the run instead.
        raise RuntimeError(
            f"Cannot load the model configuration {config_file}: {exc}"
        ) from exc

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    num_classes = configs["model"]["num_classes"] + 1
    # Import from local segformer3d.py in core package
    from .segformer3d import Segformer

    model = Segformer(
        block3d_size=nbmat_sz,
        in_chans=1,
        num_classes=num_classes,
        patch_size=configs["model"]["patch_size"],
        decoder_dim=configs["model"]["decoder_dim"],
        embed_dims=configs["model"]["channel_dims"],
        num_heads=configs["model"]["num_heads"],
        mlp_ratios=configs["model"]["MLP_ratios"],
        qkv_bias=configs["model"]["qkv_bias"],
        depths=configs["model"]["depths"],
        sr_ratios=configs["model"]["SR_ratios"],
        drop_rate=0.0,
        drop_path_rate=0.0,
    )

    # Weights are always read onto the CPU first, then the whole model
    # moves to the requested device: the one code path that works for
    # CUDA, Apple MPS and CPU alike.
    from ..utils.weights import verified_weights
    with verified_weights(model_path, "segformer3d_urbanfiltering") as weights:
        state_dict = torch.load(weights, map_location=torch.device("cpu"), weights_only=True)


    model.max_accu = state_dict.get('max_accu', 0.0)
    if 'max_accu' in state_dict:
        state_dict.pop('max_accu')

    model.best_mIoU = state_dict.get('best_mIoU', 0.0)
    if 'best_mIoU' in state_dict:
        state_dict.pop('best_mIoU')

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, configs


def predict_blocks(model, configs, device, pcd, if_bottom_only=False,
                   progress_callback=lambda x: None):
    """Voxelise ``pcd`` block by block and run the loaded SegFormer on it.

    ``device`` is ``"cuda"``, ``"mps"`` or ``"cpu"``. Returns one model
    class id per input point, with 0 meaning "no prediction".
    """
    device = resolve_device(device)
    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])
    num_classes = configs["model"]["num_classes"] + 1
    progress_callback(10)

    def _forward(x):
        try:
            with torch.no_grad():
                return model(x.to(device))
        except NotImplementedError as exc:
            if device == "mps":
                raise RuntimeError(
                    "This model uses 3D convolutions that your torch build "
                    "does not support on Apple MPS. Uncheck 'Use GPU' (or "
                    "pick CPU as the compute device) to run on the CPU."
                ) from exc
            raise

    nb_tsz = int(np.prod(nbmat_sz))

    # voxelize
    if if_bottom_only:
        cut_dim = 2
    else:
        cut_dim = 3

    _, block_idx_groups = sliding_blocks_point_indices(
        pcd[:, :cut_dim], min_res[:cut_dim] * nbmat_sz[:cut_dim], overlap_ratio=0.1)

    nb_idxs = []
    nb_pcd_idxs = []
    nb_inverse_idxs = []

    for idx in block_idx_groups:
        columns_sp = pcd[idx, :]
        nb_pcd_idx = idx

        sp_min = np.min(columns_sp[:, :3], axis=0)
        nb_ijk = np.floor((columns_sp[:, :3] - sp_min) / min_res)
        nb_sel = np.all((nb_ijk < nbmat_sz) & (nb_ijk >= 0), axis=1)
        nb_ijk = nb_ijk[nb_sel]
        nb_pcd_idx = nb_pcd_idx[nb_sel]

        nb_idx = np.ravel_multi_index(nb_ijk.astype(np.int32).T, nbmat_sz)
        nb_idx_u, nb_inverse_idx = np.unique(nb_idx, return_inverse=True)

        nb_idxs.append(nb_idx_u)  # indicies of unique voxels from each block
        # indices used to reproject the unique voxels to original order
        nb_inverse_idxs.append(nb_inverse_idx)
        # within-voxel point indices from the point cloud
        nb_pcd_idxs.append(nb_pcd_idx)

    progress_callback(15)

    # apply the DL model blockwisely
    if num_classes > 3:
        # 0 = "no prediction". A point only keeps this value if it never
        # entered any block (the grid now always covers the extent, so this
        # is a safety net) or fell outside a block's voxel grid. Model id 0
        # is absent from the class mapping, so the callers' unmapped-id
        # warning surfaces any leftover as ASPRS 0 instead of silently
        # handing it the last class (7 = Building, the previous default).
        pcd_pred = np.zeros(len(pcd), dtype=int)

        total_nbs = len(nb_idxs)
        for i in range(total_nbs):
            nb_idx = nb_idxs[i]

            x = torch.zeros(nb_tsz, 1)
            if len(nb_idx) > 0:
                x[nb_idx, :] = 1.0

            x = torch.swapaxes(
                torch.moveaxis(
                    x.reshape(
                        (1, *nbmat_sz, 1)).float(), -1, 1), -1, 2)
            h = _forward(x)

            h_nonzero = torch.moveaxis(torch.unsqueeze(torch.moveaxis(torch.swapaxes(
                h, -1, 2), 1, -1).reshape((nb_tsz, num_classes))[nb_idx, :], 0), -1, 1)
            h_nonzero = torch.argmax(h_nonzero[0], dim=0)

            nb_pred = h_nonzero.cpu().detach().numpy()
            pcd_pred[nb_pcd_idxs[i]] = nb_pred[nb_inverse_idxs[i]]

            progress_value = int(85 * i / total_nbs) + 15
            progress_callback(progress_value)

        progress_callback(100)
        return pcd_pred.astype(np.int32)  # int(1,2)

    else:  # binary case
        pcd_pred = np.full(len(pcd), dtype=bool, fill_value=False)

        total_nbs = len(nb_idxs)
        for i in range(total_nbs):
            nb_idx = nb_idxs[i]

            x = torch.zeros(nb_tsz, 1)
            if len(nb_idx) > 0:
                x[nb_idx, :] = 1.0

            x = torch.swapaxes(
                torch.moveaxis(
                    x.reshape(
                        (1, *nbmat_sz, 1)).float(), -1, 1), -1, 2)
            h = _forward(x)

            h_nonzero = torch.moveaxis(torch.unsqueeze(torch.moveaxis(torch.swapaxes(
                h, -1, 2), 1, -1).reshape((nb_tsz, num_classes))[nb_idx, :], 0), -1, 1)
            h_nonzero = torch.argmax(h_nonzero[0], dim=0)

            nb_pred = h_nonzero.cpu().detach().numpy()
            pcd_pred[nb_pcd_idxs[i]] = pcd_pred[nb_pcd_idxs[i]] | (
                nb_pred[nb_inverse_idxs[i]] > 1)  # Flag the binary over the overlapped area

            progress_value = int(85 * i / total_nbs) + 15
            progress_callback(progress_value)

        # pcd_pred[pcd_pred > 2.0] = 2.0
        if if_bottom_only:
            seen = np.zeros_like(pcd_pred, dtype=bool)
            # mark every index that was ever visited
            seen[np.concatenate(nb_pcd_idxs)] = True
            pcd_pred[~seen] = True

        progress_callback(100)
        return pcd_pred.astype(np.int32) + 1  # bool(False,True) to int(1,2)


def filterPoints(
        config_file,
        pcd,
        model_path,
        if_bottom_only=True,
        use_efficient=True,
        device="cuda",
        progress_callback=lambda x: None):
    """v1.0 entry point kept for compatibility: load the model and predict.

    ``device`` is ``"cuda"``, ``"mps"`` or ``"cpu"`` (the v1.0 booleans
    are still accepted). Returns one model class id per input point,
    with 0 meaning "no prediction".
    """
    model, configs = load_segformer(config_file, model_path, device)
    return predict_blocks(
        model, configs, device, pcd,
        if_bottom_only=if_bottom_only, progress_callback=progress_callback,
    )
