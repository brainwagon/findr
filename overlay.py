"""Draw the Overlay for a solved field.

`render` takes a solved image, the Solver backend's result, the Catalog and the
constellation name, and draws star labels and constellation boundaries onto the
image. Drawing is best-effort: failures are logged, never raised, because an
Overlay is cosmetic and must not fail a solve.
"""

import logging

import numpy as np
from PIL import ImageDraw
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points

from catalog import decode_simbad_greek

logger = logging.getLogger(__name__)


def render(image, result, catalog, constellation):
    """Draw star labels and constellation boundaries onto an image.

    Args:
        image: the PIL image to annotate; mutated and returned.
        result: the SolverResult from the solve.
        catalog: a Catalog of star names, boundaries and the label font.
        constellation: the constellation name, used to pick boundaries.

    Returns:
        The annotated image.
    """
    draw = ImageDraw.Draw(image)
    _draw_star_labels(draw, result, catalog)
    try:
        _draw_boundaries(draw, result, constellation, catalog.boundaries)
    except Exception as e:
        logger.warning("Constellation boundaries not drawn: %s", e)
    return image


def _draw_star_labels(draw, result, catalog):
    """Label each matched star with its catalogue name."""
    for star_id, point in zip(result.matched_cat_ids, result.matched_centroids):
        try:
            position = (int(point[1]) + 8, int(point[0]) - 8)
            label = decode_simbad_greek(
                catalog.star_names.get(star_id, str(star_id))
            )
            fields = label.split()
            if fields and fields[0] == "*":
                label = " ".join(fields[1:])
            draw.text(position, label, fill=(255, 255, 255), font=catalog.font)
        except Exception as e:
            logger.warning("Could not label star %s: %s", star_id, e)


def _draw_boundaries(draw, result, constellation, boundaries):
    """Project the constellation's boundary points through a fitted WCS."""
    matched_stars = np.array(result.matched_stars)
    matched_centroids = np.array(result.matched_centroids)
    if len(matched_stars) == 0 or len(matched_centroids) == 0:
        return

    star_xy = (matched_centroids[:, 1], matched_centroids[:, 0])
    world_coords = SkyCoord(
        ra=np.array(matched_stars[:, 0]) * u.deg,
        dec=np.array(matched_stars[:, 1]) * u.deg,
        frame="icrs",
    )
    wcs = fit_wcs_from_points(
        star_xy, world_coords, projection="TAN", sip_degree=2
    )

    name = (constellation or "").upper()
    if name not in boundaries:
        return

    pixel_points = []
    for ra, dec in boundaries[name]:
        try:
            px, py = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
            pixel_points.append((px, py))
        except Exception as e:
            logger.warning("Could not project boundary point: %s", e)
            pixel_points.append(None)

    for p1, p2 in zip(pixel_points, pixel_points[1:]):
        if p1 and p2:
            draw.line([p1, p2], fill="yellow", width=1)
