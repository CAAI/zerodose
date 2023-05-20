"""Visualizations for ZeroDose."""

import math

import numpy as np
import torch

from zerodose.processing import _crop_mni_to_192
from zerodose.utils import load_nifty


def _real_fake_compare_vis_metric2(  # noqa
    real, fake, mri, mask, err, errmask, save_fname
):  # noqa
    import cv2 as cv  # type: ignore
    import matplotlib  # type: ignore
    import matplotlib as mpl  # type: ignore
    from matplotlib import pyplot as plt  # type: ignore
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # type: ignore
    from numpy.ma import masked_array  # type: ignore
    from scipy import ndimage  # type: ignore

    def _get_cmap():
        colors = [
            [0, 0, 0],
            [0, 26, 255],
            [66, 245, 245],
            [66, 245, 72],
            [245, 233, 66],
            [245, 66, 66],
            [255, 255, 255],
        ]
        fracs = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 6 / 6]

        cdict = {}
        red = []
        green = []
        blue = []
        for i in range(len(colors)):
            c = colors[i][0] / 255
            red.append((fracs[i], c, c))
            c = colors[i][1] / 255
            green.append((fracs[i], c, c))
            c = colors[i][2] / 255
            blue.append((fracs[i], c, c))
        cdict = {"red": red, "green": green, "blue": blue}

        cmap = matplotlib.colors.LinearSegmentedColormap(
            "testCmap", segmentdata=cdict, N=256
        )
        return cmap

    def _crop_and_zoom(im, left=0, right=0, top=0, bottom=0):
        top, bottom = bottom, top
        x, y = im.shape
        im2 = im[bottom : x - top, left : y - right]
        frac_x = (x - (top + bottom)) / x
        frac_y = (y - (left + right)) / y
        new_x = round(x * frac_x / frac_y)
        return cv.resize(im2, (y, new_x))

    # AXIAL
    def _axial(x, _s):
        s = round((_s - (91 / 2)) * 192 / 91 + 192 / 2)
        o = np.flip(np.rot90(x[:, :, s]), axis=1)
        if _s == 60:
            o = o[16:-20, :]
        o = _crop_and_zoom(o, left=19, right=21, top=4, bottom=0)
        return o

    def _coronal(x, s):
        s = round(s * 92 / 45 - 72 / 5)
        o = np.rot90((x[:, -s, :]).T, k=2)[30:-22, :]
        o = _crop_and_zoom(o, left=21, right=23, bottom=0, top=0)
        return o

    def _saggital(x, s):
        s = round((s - (91 / 2)) * 192 / 91 * 0.98 + 192 / 2)
        o = np.rot90(np.flip(x[s, :], axis=1), k=3)[30:-10, :]
        return o

    def _imslices(x, article=False):
        is_bool = x.dtype == bool
        x = x.astype("float")

        if article:
            ax = [_axial(x, s) for s in [60, 40]]
            co = [_coronal(x, s) for s in [50]]
            sa = [_saggital(x, s) for s in [50]]
        else:
            ax = [_axial(x, s) for s in [60, 50, 40, 30]]
            co = [_coronal(x, s) for s in [40, 50]]
            sa = [_saggital(x, s) for s in [30, 50]]
        out = np.concatenate(ax + co + sa, axis=0)
        if is_bool:
            return out > 0.2
        return out

    cmap = _get_cmap()
    article = True
    min_pet = 0
    max_pet = math.ceil(np.quantile(fake[mask], 0.99))
    min_abn = -60
    max_abn = 60
    min_mr = math.ceil(np.quantile(mri[mask], 0.01))
    max_mr = math.ceil(np.quantile(mri[mask], 0.99))
    cbar_offset = -0.09
    _cbar_width = 0.21
    cbar_width = f"{_cbar_width*100}%"
    _cbar_height = 0.020
    cbar_height = f"{_cbar_height*100}%"

    cbar_spacing = (1 - _cbar_width * 4) / 4
    cbar_const = cbar_spacing / 2
    tick_size = 15
    cbar_label_size = 18
    label_offset = 1.8

    # RD Edge:
    mask = mask > 0.01
    struct = ndimage.generate_binary_structure(3, 3)
    edges = ndimage.binary_dilation(
        ndimage.binary_dilation(ndimage.binary_dilation(mask, struct))
    )
    edges = edges & ~mask

    nedge = np.zeros(edges.shape)
    nedge[1:-1, 1:-1, 1:-1] = edges[1:-1, 1:-1, 1:-1]
    nedge = _imslices(nedge, article=article)
    nedge[nedge != 0] = 0.1

    # MRI
    im = _imslices(mri, article=article)
    mask = _imslices(mask, article=article)

    total_im = im
    total_mask = mask

    mask_ = np.zeros(total_im.shape)

    # FAKE
    im = _imslices(fake, article=article)

    total_im = np.concatenate((total_im, im), axis=1)
    total_mask = np.concatenate((total_mask, mask), axis=1)

    # Real

    im = _imslices(real, article=article)

    total_im = np.concatenate((total_im, im), axis=1) if total_im is not None else im
    mask_ = np.concatenate((mask_, np.ones(im.shape), np.ones(im.shape)), axis=1)

    total_mask = (
        np.concatenate((total_mask, mask), axis=1) if total_mask is not None else mask
    )

    # RD

    im_rel = _imslices(err, article=article)
    total_im = np.concatenate((total_im, im_rel), axis=1)
    total_mask = np.concatenate(
        (total_mask, _imslices(errmask, article=article)), axis=1
    )

    mask_rd = np.ones(im.shape) * 2
    # mask_rd[nedge > 0] = 3
    mask_ = np.concatenate((mask_, mask_rd), axis=1)
    mask_rd = np.ones(im.shape) * 4
    # mask_ = np.concatenate((mask_, mask_rd), axis=1)

    # Params
    params = {
        "ytick.color": "w",
        "xtick.color": "w",
        "axes.labelcolor": "w",
        "axes.edgecolor": "w",
    }

    f, (a0) = plt.subplots(1, 1, figsize=(12, 13))

    plt.rcParams.update(params)
    ax = plt.gca()
    ax.set_facecolor("black")
    f.patch.set_facecolor("black")
    a0.axis("off")

    v1a = masked_array(total_im, mask_ != 0)
    v1b = masked_array(total_im, mask_ != 1)
    v1c = masked_array(total_im, mask_ != 2)
    v1d = masked_array(total_im, mask_ != 5)

    a0.imshow(
        v1a,
        cmap="gray",
        interpolation="nearest",
        vmin=min_mr,
        vmax=max_mr,
        alpha=total_mask.astype(float),
    )
    a0.imshow(
        v1b,
        cmap="Spectral_r",
        interpolation="nearest",
        vmin=min_pet,
        vmax=max_pet,
        alpha=total_mask.astype(float),
    )

    a0.imshow(
        v1c,
        cmap=cmap,
        alpha=total_mask.astype(float),
        interpolation="nearest",
        vmin=min_abn,
        vmax=max_abn,
    )
    a0.imshow(
        v1d,
        cmap="gray",
        interpolation="nearest",
        vmin=min_mr,
        vmax=max_mr,
        alpha=total_mask.astype(float),
    )

    norm = mpl.colors.Normalize(vmin=min_abn, vmax=max_abn)

    # Colorbars
    axins = inset_axes(
        a0,
        width=cbar_width,  # width = 5% of parent_bbox width
        height=cbar_height,  # height : 50%
        loc="lower left",
        bbox_to_anchor=(
            cbar_const + 3 * (_cbar_width + cbar_spacing),
            cbar_offset,
            1,
            1,
        ),
        bbox_transform=a0.transAxes,
        borderpad=0,
    )
    cb1 = mpl.colorbar.ColorbarBase(
        axins,
        cmap=cmap,
        orientation="horizontal",
        norm=norm,
        ticks=[min_abn, 0, max_abn],
    )
    cb1.ax.tick_params(labelsize=tick_size)

    plt.text(
        0.5,
        label_offset,
        "Abn. map [%]",
        transform=axins.transAxes,
        c="white",
        ha="center",
        fontsize=cbar_label_size,
    )
    plt.gcf().add_axes(axins)

    norm = mpl.colors.Normalize(vmin=min_pet, vmax=max_pet)

    axins = inset_axes(
        a0,
        width=cbar_width,  # width = 5% of parent_bbox width
        height=cbar_height,  # height : 50%
        loc="lower left",
        bbox_to_anchor=(
            cbar_const + 1 * (_cbar_width + cbar_spacing),
            cbar_offset,
            1,
            1,
        ),
        bbox_transform=a0.transAxes,
        borderpad=0,
    )
    cb1 = mpl.colorbar.ColorbarBase(
        axins,
        cmap=mpl.cm.Spectral_r,
        orientation="horizontal",
        norm=norm,
        ticks=[min_pet, max_pet],
    )
    cb1.ax.tick_params(labelsize=tick_size)

    plt.gcf().add_axes(axins)
    plt.text(
        0.5,
        label_offset,
        "sbPET [kBq/ml]",
        transform=axins.transAxes,
        c="white",
        ha="center",
        fontsize=cbar_label_size,
    )

    axins = inset_axes(
        a0,
        width=cbar_width,  # width = 5% of parent_bbox width
        height=cbar_height,  # height : 50%
        loc="lower left",
        bbox_to_anchor=(
            cbar_const + 2 * (_cbar_width + cbar_spacing),
            cbar_offset,
            1,
            1,
        ),
        bbox_transform=a0.transAxes,
        borderpad=0,
    )
    cb1 = mpl.colorbar.ColorbarBase(
        axins,
        cmap=mpl.cm.Spectral_r,
        orientation="horizontal",
        norm=norm,
        ticks=[min_pet, max_pet],
    )
    cb1.ax.tick_params(labelsize=tick_size)

    plt.text(
        0.5,
        label_offset,
        "PET [kBq/ml]",
        transform=axins.transAxes,
        c="white",
        ha="center",
        fontsize=cbar_label_size,
    )
    plt.gcf().add_axes(axins)

    norm = mpl.colors.Normalize(vmin=min_mr, vmax=max_mr)

    axins = inset_axes(
        a0,
        width=cbar_width,  # width = 5% of parent_bbox width
        height=cbar_height,  # height : 50%
        loc="lower left",
        bbox_to_anchor=(cbar_const, cbar_offset, 1, 1),
        bbox_transform=a0.transAxes,
        borderpad=0,
    )
    cb1 = mpl.colorbar.ColorbarBase(
        axins,
        cmap=mpl.cm.gray,
        orientation="horizontal",
        norm=norm,
        ticks=[round(min_mr), round(max_mr)],
    )

    plt.text(
        0.5,
        label_offset,
        "MRI",
        transform=axins.transAxes,
        c="white",
        ha="center",
        fontsize=cbar_label_size,
    )

    cb1.ax.tick_params(labelsize=tick_size)
    plt.gcf().add_axes(axins)
    plt.savefig(save_fname, dpi=300, bbox_inches="tight")


def create_article_figure(
    mri_fname, pet_fname, sbpet_fname, mask_fname, abn_fname, save_fname
):
    """Create the article figure."""
    mask = load_nifty(mask_fname)
    fake = load_nifty(sbpet_fname)
    rd = load_nifty(abn_fname)
    mr = load_nifty(mri_fname)
    real = load_nifty(pet_fname)

    real = (_crop_mni_to_192(real.unsqueeze(0)) / 1000).squeeze()
    fake = (_crop_mni_to_192(fake.unsqueeze(0)) / 1000).squeeze()
    rd = _crop_mni_to_192(rd.unsqueeze(0)).squeeze()
    mr = _crop_mni_to_192(mr.unsqueeze(0)).squeeze()
    mask = _crop_mni_to_192((mask.unsqueeze(0)) > 0.5).type(torch.bool).squeeze()

    _real_fake_compare_vis_metric2(
        real.numpy(),
        fake.numpy(),
        mr.numpy(),
        mask.numpy(),
        rd.numpy(),
        mask.numpy(),
        save_fname,
    )


if __name__ == "__main__":
    mask = (
        "/home/hinge/Projects/zerodose/tests/mni_test_data/"
        "mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
    )
    fake = "/home/hinge/Projects/zerodose/tests/mni_test_data/mni_sbpet.nii.gz"
    rd = "/home/hinge/Projects/zerodose/tests/mni_test_data/abn.nii.gz"
    mr = (
        "/home/hinge/Projects/zerodose/tests/mni_test_data/mni_icbm152_nlin_sym_09a/"
        "mni_icbm152_t1_tal_nlin_sym_09a.nii"
    )
    real = "/home/hinge/Projects/zerodose/tests/mni_test_data/MNI152_PET_1mm.nii"

    create_article_figure(
        mri_fname=mr,
        pet_fname=real,
        sbpet_fname=real,
        mask_fname=mask,
        abn_fname=real,
        save_fname="/home/hinge/Projects/zerodose/tests/mni_test_data/figure1.png",
    )
