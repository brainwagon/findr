"""Draw the Overlay for a solved field.

`render` takes a solved image, the Solver backend's result, the Catalog and the
constellation name, and draws star labels and constellation boundaries onto the
image. Drawing is best-effort: failures are logged, never raised, because an
Overlay is cosmetic and must not fail a solve.
"""

import itertools
import logging

import numpy as np
from PIL import ImageDraw
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points

from catalog import decode_simbad_greek

logger = logging.getLogger(__name__)

# Labelling every matched star and fitting the boundary WCS from all of them
# costs more than it shows: a solve can match hundreds of stars. Cap the labels
# and fit the projection from a small bright subset instead.
MAX_LABELLED_STARS = 30
MAX_WCS_POINTS = 8


def render(image, result, catalog, constellation, boundaries=True):
    """Draw star labels, and optionally constellation boundaries, on an image.

    Args:
        image: the PIL image to annotate; mutated and returned.
        result: the SolverResult from the solve.
        catalog: a Catalog of star names, boundaries and the label font.
        constellation: the constellation name, used to pick boundaries.
        boundaries: when False, skip the (expensive) boundary drawing.

    Returns:
        The annotated image.
    """
    draw_labels(image, result, catalog)
    if boundaries:
        draw_boundaries(image, result, constellation, catalog.boundaries)
    return image


def draw_labels(image, result, catalog):
    """Label each matched star with its catalogue name."""
    _draw_star_labels(ImageDraw.Draw(image), result, catalog)
    return image


def draw_boundaries(image, result, constellation, boundaries):
    """Draw the constellation's boundary lines, best-effort.

    Failures are logged, never raised, because an Overlay is cosmetic and must
    not fail a solve.
    """
    try:
        _draw_boundaries(
            ImageDraw.Draw(image), result, constellation, boundaries
        )
    except Exception as e:
        logger.warning("Constellation boundaries not drawn: %s", e)
    return image


def _draw_star_labels(draw, result, catalog):
    """Label each matched star with its catalogue name."""
    labels = zip(result.matched_cat_ids, result.matched_centroids)
    for star_id, point in itertools.islice(labels, MAX_LABELLED_STARS):
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

    subset = min(len(matched_stars), MAX_WCS_POINTS)
    star_xy = (matched_centroids[:subset, 1], matched_centroids[:subset, 0])
    world_coords = SkyCoord(
        ra=np.array(matched_stars[:subset, 0]) * u.deg,
        dec=np.array(matched_stars[:subset, 1]) * u.deg,
        frame="icrs",
    )
    wcs = fit_wcs_from_points(
        star_xy, world_coords, projection="TAN", sip_degree=0
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
