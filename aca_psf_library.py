import os
import numpy as np
from collections import defaultdict
from itertools import chain
from math import floor

from astropy.table import Table
import astropy.stats
from mica.archive import aca_l0
from kadi import events
from mica.stats import guide_stats

# The centroids we want are between the middle of the 4th pixel [idx 3] and middle of the
# 5th pixel [idx 4]
LOW = 3
HI = 4
# Sampling this in 10 x 10 bins
NBINS = 10

EIGHT_LABELS = np.array([['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1'],
                         ['I1', 'J1', 'K1', 'L1', 'M1', 'N1', 'O1', 'P1'],
                         ['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2'],
                         ['I2', 'J2', 'K2', 'L2', 'M2', 'N2', 'O2', 'P2'],
                         ['A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3'],
                         ['I3', 'J3', 'K3', 'L3', 'M3', 'N3', 'O3', 'P3'],
                         ['A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4'],
                         ['I4', 'J4', 'K4', 'L4', 'M4', 'N4', 'O4', 'P4']])


BGDPIX = ['A1', 'B1', 'G1', 'H1', 'I4', 'J4', 'O4', 'P4']


class AcaPsfLibrary(object):
    drc = 0.1  # Height/width of PSF bin in pixels

    def __init__(self):
        psfs = {}

        filename = os.path.join(os.path.dirname(__file__), 'aca_psf_lib.dat')
        dat = Table.read(filename, format='ascii.basic', guess=False)
        self.dat = dat

        for row in dat:
            ii = row['row_bin_idx']
            jj = row['col_bin_idx']
            psf = np.array([row[label] for label in chain(*EIGHT_LABELS)]).reshape(8, 8)
            psfs[ii, jj] = psf

        self.psfs = psfs

    def get_psf(self, row, col, pix_zero_loc='center'):
        """
        Get interpolated ACA PSF that corresponds to pixel location ``row``, ``col``.

        :param row: float row value of PSF centroid
        :param col: float col value of PSF centroid
        :param pix_zero_loc: row/col coords are integral at 'edge' or 'center'

        :returns: 8x8 PSF image normalized to 1.0
        """
        drc = self.drc

        if pix_zero_loc == 'center':
            # Transform to 'edge' coordinates (pixel lower-left corner at 0.0, 0.0)
            row = row + 0.5
            col = col + 0.5
        elif pix_zero_loc != 'edge':
            raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

        # Subpixel position in range (-0.5, 0.5)
        r = row - round(row)
        c = col - round(col)

        # Floating point index into PSF library in range (0.0, 10.0)
        # (assuming 10x10 grid of PSFs covering central pixel
        ix = (r + 0.5) / drc - 0.5
        iy = (c + 0.5) / drc - 0.5

        # Int index into PSF library
        ii = int(floor(ix))
        jj = int(floor(iy))

        # Following wikipedia notation (Unit Square section of
        # https://en.wikipedia.org/wiki/Bilinear_interpolation)

        # Float index within subpixel bin in range (0, 1)
        x = ix - ii
        y = iy - jj

        # Finally the bilinear interpolation of the PSF images.
        f = self.psfs
        b0 = (1 - x) * (1 - y)
        b1 = x * (1 - y)
        b2 = (1 - x) * y
        b3 = x * y
        P0 = f[ii, jj]
        P1 = f[ii + 1, jj]
        P2 = f[ii, jj + 1]
        P3 = f[ii + 1, jj + 1]
        psf = P0 * b0 + P1 * b1 + P2 * b2 + P3 * b3

        return psf


# Local copy of annie centroid_fm
def prep_6x6(img, bgd=None):
    """
    Subtract background and in case of 8x8 image
    cut and return the 6x6 inner section.
    """
    # Cast to an ndarray (without copying)
    img = img.view(np.ndarray)

    if isinstance(bgd, np.ndarray):
        bgd = bgd.view(np.ndarray)

    if bgd is not None:
        img = img - bgd

    if img.shape == (8, 8):
        img = img[1:7, 1:7]

    return img


def centroid_fm(img, bgd=None, pix_zero_loc='center', norm_clip=None):
    """
    First moment centroid of ``img``.

    Return FM centroid in coords where lower left pixel of image has value
    (0.0, 0.0) at the center (for pix_zero_loc='center') or the lower-left edge
    (for pix_zero_loc='edge').

    :param img: NxN ndarray
    :param bgd: background to subtract, float of NXN ndarray
    :param pix_zero_loc: row/col coords are integral at 'edge' or 'center'
    :param norm_clip: clip image norm at this min value (default is None and
                      implies Exception for non-positive norm)

    :returns: row, col, norm float
    """

    sz_r, sz_c = img.shape
    if sz_r != sz_c:
        raise ValueError('input img must be square')

    rw, cw = np.mgrid[1:7, 1:7] if sz_r == 8 else np.mgrid[0:sz_r, 0:sz_r]

    if sz_r in (6, 8):
        img = prep_6x6(img, bgd)
        img[[0, 0, 5, 5], [0, 5, 0, 5]] = 0

    norm = np.sum(img)
    if norm_clip is not None:
        norm = norm.clip(norm_clip, None)
    else:
        if norm <= 0:
            raise ValueError('non-positive image norm {}'.format(norm))

    row = np.sum(rw * img) / norm
    col = np.sum(cw * img) / norm

    if pix_zero_loc == 'edge':
        # Transform row/col values from 'center' convention (as returned
        # by centroiding) to the 'edge' convention requested by user.
        row = row + 0.5
        col = col + 0.5
    elif pix_zero_loc != 'center':
        raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

    return row, col, norm


# generalize this so we can use gaussian later
centroid = centroid_fm


def obs_slot_psf(obsid, slot):
    """
    Generate mean psf images corresponding to a 10x10 grid of centroids in the central pixel
    of the 8x8 image
    """
    # Should we cut off the beginning and end of the dwell?
    dwell = events.dwells.filter(obsid=obsid)[0]
    images = aca_l0.get_l0_images(dwell.start, dwell.stop, slot=slot, imgsize=[8])

    psf_map = defaultdict(list)
    bin_edges = np.linspace(LOW, HI, NBINS + 1)
    mid = (LOW + HI ) / 2.
    rs = []
    cs = []
    for img in images:
        r, c, norm = centroid(np.array(img))
        # Skip any that fall outside the inner half-pixel box
        if (abs(r - mid) >= .5) or (abs(c - mid) >= .5):
            continue
        # Save these for distribution map
        rs.append(r)
        cs.append(c)
        # Not sure about the best way to index these, but digitize give the
        # in index of the right edge, so subtracting 1 for now to give range 0 to 9
        rb = np.digitize([r], bin_edges)[0] - 1
        cb = np.digitize([c], bin_edges)[0] - 1
        psf_map[(rb, cb)].append(img)

    #Bin the centroid locations in the middle up in a 10x10 array just to see distribution
    H, xedges, yedges = np.histogram2d(rs, cs, bins=[NBINS, NBINS], range=[[LOW, HI], [LOW, HI]])

    # Make psf images
    psf_images = {}
    for loc in psf_map:
        # N x 64 stack of pixels
        loc_images = np.array(psf_map[loc])
        if len(loc_images) < 10:
            raise ValueError
        # use the sum over two axes to sum up all the pixels in each 8x8 image
        norms = np.sum(loc_images, axis=(1, 2)).clip(1, None)
        # clip to just use images within 1% of median sum
        # This would be too restrictive on anything but bright star data
        loc_images = loc_images[(abs(norms - np.median(norms)) / np.median(norms)) < .01]
        # May also decide to only use N images or other filters 
        # Also not sure if we want the mean or the median pixel value for each pixel at this point, but
        # Sigma clip these in pixel stacks (the axis=0 bit)
        loc_images = astropy.stats.sigma_clip(loc_images, axis=0, sigma=2)
        # Are enough samples left around?
        if np.any(np.sum(~loc_images.mask, axis=0) < 5):
            raise ValueError
        # Get the mean image of the sigma-clip/masked data, throw out the mask 
        mean_image = np.mean(loc_images, axis=0).data
        # Normalize the mean image to one
        psf_images[loc] = mean_image / np.sum(mean_image)

    return psf_images, H.astype(int), rs, cs


def get_obs_slots():
    """
    Use the guide star database to get a Table of long-ish ER observations with bright stars tracked well
    I've used the old n100_warm_frac as a proxy for expected low-ish dark current, though the residuals
    probably support this just as well.  This doesn't check for dither-disabled explicitly; I'm hoping we'd be
    sensitive to that via the check that there is good centroid coverage within the observation.
    """
    gs = Table(guide_stats.get_stats())
    gs['dur'] = gs['npnt_tstop'].astype(float) - gs['kalman_tstart']
    ok = ((gs['obsid'] > 38000)
          & (gs['dur'] > 26000)
          & (gs['sz'] == '8x8')
          & (gs['aoacmag_mean'] < 6.5)
          & (gs['f_track'] > .99)
          & (gs['dy_std'] < .2)
          & (gs['dz_std'] < .2)
      & (gs['n100_warm_frac'] < .10))
    return gs[ok]


def combine_obs_psfs(psfs):
    # Get the mean of the means on these PSF images (do these need to be sigma clipped too?)
    psf = {}
    for loc in psfs[0]:
        new_img = np.mean(np.array([p[loc] for p in psfs]), axis=0)
        # And I think since this was the pixel stack mean, we need to renormalize
        psf[loc] = new_img / np.sum(new_img)
    return psf


def make_library(guide_stars):

    multi_obs_psfs = []
    distributions = []
    row_centroids = []
    col_centroids = []
    for obs in guide_stars:
        try:
            slot_psf, distribution, rs, cs = obs_slot_psf(obs['obsid'], obs['slot'])
        except ValueError:
            print("Not using {} {}".format(obs['obsid'], obs['slot']))
            continue
        multi_obs_psfs.append(slot_psf)
        distributions.append(distribution)
        row_centroids.append(rs)
        col_centroids.append(cs)

    if len(multi_obs_psfs) <= 1:
        return slot_psf, distributions, row_centroids, col_centroids
    master_psf = combine_obs_psfs(multi_obs_psfs)
    return master_psf, distributions, row_centroids, col_centroids


def make_psf_table(psf):
    rows = []
    for loc in sorted(psf):
        row = {'row_bin_idx': loc[0],
               'col_bin_idx': loc[1],
               'row_bin_left_edge': -0.5 + (loc[0] / NBINS),
               'row_bin_right_edge': -0.5 + (loc[0] / NBINS) + (1 / NBINS),
               'col_bin_left_edge': -0.5 + (loc[1] / NBINS),
               'col_bin_right_edge': -0.5 + (loc[1] / NBINS) + (1 / NBINS)}
        for pix_label in EIGHT_LABELS.flatten():
            row[pix_label] = psf[loc][EIGHT_LABELS == pix_label][0]
        rows.append(row)
    cols = ['row_bin_idx', 'col_bin_idx',
            'row_bin_left_edge', 'row_bin_right_edge',
            'col_bin_left_edge', 'col_bin_right_edge']
    # Put these in the iteration order
    cols.extend(EIGHT_LABELS.flatten().tolist())
    table = Table(rows)[cols]
    for col in ['row_bin_left_edge', 'row_bin_right_edge',
                'col_bin_left_edge', 'col_bin_right_edge']:
        table[col].format = "3.1f"
    for col in EIGHT_LABELS.flatten().tolist():
        table[col].format = "8.6f"
    return table


def table_to_psf(t):
    psf = {}
    for row in t:
        img = np.zeros((8, 8))
        for pix_label in EIGHT_LABELS.flatten():
            img[EIGHT_LABELS == pix_label] = row[pix_label]
        psf[(row['row_bin_idx'], row['col_bin_idx'])] = img
    return psf


def pad_psf_library(psfs):
    """Pad the PSF library by an extra bin around the edge.

    This uses the corresponding "translated" PSF from 1.0 pixels over.  To
    generate the pixel at r, c = -0.05, 0.5, use the one from 0.95, 0.5 and then
    shift the PSF image pixels by 1 pixel in row.
    """
    def get_slices(idx):
        if idx == -1:
            slice_in = slice(1, None)
            slice_out = slice(None, -1)
        elif idx == NBINS:
            slice_in = slice(None, -1)
            slice_out = slice(1, None)
        else:
            slice_in = slice(None, None)
            slice_out = slice(None, None)
        return slice_in, slice_out

    for ii_out in range(-1, NBINS + 1):
        ii_in = np.mod(ii_out, NBINS)
        ii_slice_in, ii_slice_out = get_slices(ii_out)

        for jj_out in range(-1, NBINS + 1):
            if (ii_out, jj_out) in psfs:
                continue

            jj_in = np.mod(jj_out, NBINS)
            jj_slice_in, jj_slice_out = get_slices(jj_out)

            psf = np.zeros(shape=(8, 8), dtype=float)
            psf[ii_slice_out, jj_slice_out] = psfs[ii_in, jj_in][ii_slice_in, jj_slice_in]
            psfs[ii_out, jj_out] = psf


def create_psf_library():
    """Make and write out a library"""

    guide_stars = get_obs_slots()
    psf, distrib, row_cen, col_cen = make_library(guide_stars)
    pad_psf_library(psf)
    table = make_psf_table(psf)
    table.write("aca_psf_lib.dat", format='ascii', overwrite=True)
