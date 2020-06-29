import numpy as np
import pytest

from gsm.models.gaussian import from_los as gaussian_from_los


def test__analytical():
    mean = lambda x: return 0
    scale = lambda x: return 1

    gaussian_from_los.losmoments2gaussian(mean, scale)
    
