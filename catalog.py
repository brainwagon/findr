"""Reference data for labelling a solved field.

The Catalog holds the star identifiers, constellation boundaries and label font
used to annotate a solved image. It is distinct from a Solver backend's own star
database, which the backend uses to match the star pattern.
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from PIL import ImageFont

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_IDS_PATH = os.path.join(BASE_DIR, "ids.csv")
DEFAULT_BOUNDARIES_PATH = os.path.join(BASE_DIR, "bound_20.dat")
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSansDisplay-Regular.ttf"
DEFAULT_FONT_SIZE = 12

_GREEK_MAP = {
    'alf': 'α', 'bet': 'β', 'gam': 'γ', 'del': 'δ', 'eps': 'ε', 'zet': 'ζ',
    'eta': 'η', 'tet': 'θ', 'iot': 'ι', 'kap': 'κ', 'lam': 'λ', 'mu.': 'μ',
    'nu.': 'ν', 'ksi': 'ξ', 'omi': 'ο', 'pi.': 'π', 'rho': 'ρ', 'sig': 'σ',
    'tau': 'τ', 'ups': 'υ', 'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'ome': 'ω',
}


@dataclass(frozen=True)
class Catalog:
    """Star identifiers, constellation boundaries and label font."""

    star_names: Dict[int, str] = field(default_factory=dict)
    boundaries: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    font: object = None


def decode_simbad_greek(text):
    """Replace Simbad's three-letter Greek codes with Greek characters."""
    result = text
    for code, greek in _GREEK_MAP.items():
        result = result.replace(code, greek)
    return result


def _load_star_names(path):
    names = {}
    with open(path, "r") as f:
        for row in csv.reader(f):
            star_id, bayer, proper = row
            names[int(star_id)] = bayer if bayer else proper
    return names


def _load_boundaries(path):
    boundaries = {}
    with open(path, "r") as f:
        for line in f:
            ra_h = float(line[0:10])
            dec_d = float(line[11:22])
            constellation = line[23:27].strip()
            boundaries.setdefault(constellation, []).append((ra_h * 15.0, dec_d))
    return boundaries


def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        logger.warning("Font %s not found; using the default bitmap font.", path)
        return ImageFont.load_default()


def load_catalog(
    ids_path=DEFAULT_IDS_PATH,
    boundaries_path=DEFAULT_BOUNDARIES_PATH,
    font_path=DEFAULT_FONT_PATH,
    font_size=DEFAULT_FONT_SIZE,
):
    """Load the star names, boundaries and font into a Catalog.

    Missing files are tolerated: the catalog simply lacks that data, so a
    solved field can still be returned without annotations.
    """
    star_names = {}
    if os.path.exists(ids_path):
        star_names = _load_star_names(ids_path)
    else:
        logger.warning("Star identifiers %s not found.", ids_path)

    boundaries = {}
    if os.path.exists(boundaries_path):
        boundaries = _load_boundaries(boundaries_path)
    else:
        logger.warning("Constellation boundaries %s not found.", boundaries_path)

    return Catalog(
        star_names=star_names,
        boundaries=boundaries,
        font=_load_font(font_path, font_size),
    )
