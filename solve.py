"""The plate-solve pipeline.

`run_solve` turns a captured image into a Solve outcome: it asks a Solver
backend to identify the field, enriches the result with site-relative altitude
and azimuth and the constellation, annotates the image, and returns everything
as a `SolveOutcome`. Its dependencies are injected, so it can be tested without
a camera or a star database.

Image acquisition sits behind the `ImageSource` seam: `TestImageSource` for Test
mode, `CameraImageSource` for the live camera. The `SolveStore` holds the status
of the current solve and its outcome for the web layer.
"""

import io
import logging
import math
import os
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import ephem
import numpy as np
from PIL import Image, ImageDraw
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points

from catalog import decode_simbad_greek

logger = logging.getLogger(__name__)


class ImageSourceError(Exception):
    """Raised when an Image source cannot provide an image."""


@dataclass(frozen=True)
class SolveOutcome:
    """The complete result of one plate solve."""

    ra: Optional[float] = None
    dec: Optional[float] = None
    roll: Optional[float] = None
    fov: Optional[float] = None
    matched_stars_count: int = 0
    ra_hms: str = ""
    dec_dms: str = ""
    alt: Optional[float] = None
    az: Optional[float] = None
    constellation: str = ""
    image: Optional[Image.Image] = None
    error: Optional[str] = None

    @property
    def ok(self):
        """True when the solve produced coordinates."""
        return self.error is None

    def to_json(self):
        """The outcome as the keys the web layer has always emitted."""
        if not self.ok:
            return {"error": self.error}
        return {
            "ra": f"{self.ra:.4f}",
            "dec": f"{self.dec:.4f}",
            "roll": f"{self.roll:.4f}",
            "ra_hms": self.ra_hms,
            "dec_dms": self.dec_dms,
            "alt": f"{self.alt:.1f}",
            "az": f"{self.az:.1f}",
            "constellation": self.constellation,
            "matched_stars_count": self.matched_stars_count,
        }


class ImageSource(ABC):
    """A seam over where the image to be solved comes from."""

    @abstractmethod
    def acquire(self):
        """Return a PIL image, or raise ImageSourceError."""
        raise NotImplementedError


class TestImageSource(ImageSource):
    """Acquire a random image from the pre-loaded test-image set."""

    def __init__(self, directory="test-images", rng=None):
        self._directory = directory
        self._rng = rng if rng is not None else random.Random()

    def acquire(self):
        try:
            files = [
                f for f in os.listdir(self._directory)
                if f.lower().endswith((".jpg", ".jpeg"))
            ]
        except OSError as e:
            raise ImageSourceError(f"Test images unavailable: {e}")
        if not files:
            raise ImageSourceError("No test images found.")
        return _load_image(os.path.join(self._directory, self._rng.choice(files)))


class CameraImageSource(ImageSource):
    """Acquire a lores frame from the camera."""

    def __init__(self, camera, name="lores"):
        self._camera = camera
        self._name = name

    def acquire(self):
        buffer = io.BytesIO()
        self._camera.capture_file(buffer, name=self._name, format="jpeg")
        buffer.seek(0)
        return _load_image(buffer)


def _load_image(source):
    image = Image.open(source)
    image.load()
    return image


def format_radec_fixed_width(angle_obj, is_ra=True, total_width=10, decimal_places=1):
    """Format an ephem Angle to a fixed-width string.

    RA: HH:MM:SS.S (total_width=10)
    Dec: sDD:MM:SS.S (total_width=11, s is sign)
    """
    parts = str(angle_obj).split(":")

    if is_ra:
        hours = parts[0].zfill(2)
        minutes = parts[1].zfill(2)
        seconds = f"{float(parts[2]):0{3 + decimal_places}.{decimal_places}f}"
        formatted = f"{hours}:{minutes}:{seconds}"
    else:
        sign = ""
        if parts[0].startswith("-"):
            sign = "-"
            parts[0] = parts[0][1:]
        elif parts[0].startswith("+"):
            sign = "+"
            parts[0] = parts[0][1:]
        degrees = parts[0].zfill(2)
        minutes = parts[1].zfill(2)
        seconds = f"{float(parts[2]):0{3 + decimal_places}.{decimal_places}f}"
        formatted = f"{sign}{degrees}:{minutes}:{seconds}"

    return formatted.ljust(total_width)[:total_width]


def run_solve(image, solver, observer, catalog, clock=ephem.now):
    """Solve one image and return a SolveOutcome.

    Args:
        image: the PIL image to solve.
        solver: a BaseSolver (or SolverManager) that identifies the field.
        observer: an ephem.Observer giving the observing site.
        catalog: a Catalog used to label stars and draw boundaries.
        clock: a callable returning the observation time; injected for tests.

    Returns:
        A SolveOutcome; `error` is set when no solution was found.
    """
    try:
        result = solver.solve(image)
    except Exception as e:
        logger.error("Solver raised during solve: %s", e)
        return SolveOutcome(error=str(e), image=image)

    if not result:
        logger.warning("Plate solve failed to find a solution.")
        return SolveOutcome(error="No solution found.", image=image)

    ra_val = result.ra
    dec_val = result.dec
    roll_val = result.roll
    fov_val = result.fov

    ra_hms = ephem.hours(math.radians(ra_val))
    dec_dms = ephem.degrees(math.radians(dec_val))

    observer.date = clock()
    target = ephem.FixedBody()
    target._ra = ra_hms
    target._dec = dec_dms
    target.compute(observer)

    constellation = ephem.constellation(
        (math.radians(ra_val), math.radians(dec_val))
    )[0]

    outcome = SolveOutcome(
        ra=ra_val,
        dec=dec_val,
        roll=roll_val,
        fov=fov_val,
        matched_stars_count=result.matched_stars_count,
        ra_hms=format_radec_fixed_width(
            ra_hms, is_ra=True, total_width=10, decimal_places=1
        ),
        dec_dms=format_radec_fixed_width(
            dec_dms, is_ra=False, total_width=11, decimal_places=1
        ),
        alt=math.degrees(target.alt),
        az=math.degrees(target.az),
        constellation=constellation,
        image=image,
    )

    _annotate(image, result, constellation, catalog)
    return outcome


def _annotate(image, result, constellation, catalog):
    """Draw star labels and constellation boundaries onto the image."""
    draw = ImageDraw.Draw(image)
    matched_cat_ids = result.matched_cat_ids
    matched_centroids = result.matched_centroids

    for star_id, point in zip(matched_cat_ids, matched_centroids):
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

    try:
        _draw_boundaries(draw, result, constellation, catalog.boundaries)
    except Exception as e:
        logger.warning("Constellation boundaries not drawn: %s", e)


def _draw_boundaries(draw, result, constellation, boundaries):
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
        except Exception:
            pixel_points.append(None)

    for p1, p2 in zip(pixel_points, pixel_points[1:]):
        if p1 and p2:
            draw.line([p1, p2], fill="yellow", width=1)


class SolveStore:
    """The status of the current solve and its Solve outcome, under a lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = "idle"
        self._outcome = None
        self._image_bytes = None

    def begin(self):
        """Mark a new solve as in progress."""
        with self._lock:
            self._status = "solving"
            self._outcome = None

    def set_status(self, status):
        """Set the status without an outcome (e.g. paused)."""
        with self._lock:
            self._status = status

    def finish(self, outcome, image_bytes=None):
        """Store a finished outcome and its encoded image."""
        with self._lock:
            self._outcome = outcome
            self._status = "solved" if outcome.ok else "failed"
            if image_bytes is not None:
                self._image_bytes = image_bytes

    def snapshot(self):
        """Return (status, outcome) atomically."""
        with self._lock:
            return self._status, self._outcome

    def get_image_bytes(self):
        """Return the JPEG bytes of the most recent solve, if any."""
        with self._lock:
            return self._image_bytes
