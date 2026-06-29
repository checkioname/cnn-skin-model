import os
import sys
import numpy as np
from PIL import Image
import pytest
from hypothesis import given, strategies as st, settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.preprocessing.PreProcessing import OpenCVPreprocessing


class TestOpenCVPreprocessing:
    def test_output_is_pil_rgb(self):
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = OpenCVPreprocessing()(img)
        assert result.mode == "RGB"
        assert isinstance(result, Image.Image)

    def test_output_size_matches_input(self):
        img = Image.new("RGB", (224, 224), color=(64, 128, 192))
        result = OpenCVPreprocessing()(img)
        assert result.size == (224, 224)

    @given(st.integers(min_value=50, max_value=500), st.integers(min_value=50, max_value=500))
    @settings(max_examples=10)
    def test_various_sizes(self, w, h):
        img = Image.new("RGB", (w, h), color=(100, 150, 200))
        result = OpenCVPreprocessing()(img)
        assert result.size == (w, h)
        assert result.mode == "RGB"


class TestTransforms:
    def test_resize_center_crop_512(self):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
        ])
        img = Image.new("RGB", (4000, 2700), color=(255, 0, 0))
        result = transform(img)
        assert result.size == (512, 512)

    def test_resize_center_crop_square(self):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
        ])
        img = Image.new("RGB", (1024, 1024), color=(0, 255, 0))
        result = transform(img)
        assert result.size == (512, 512)

    @given(st.integers(min_value=600, max_value=2000), st.integers(min_value=600, max_value=2000))
    @settings(max_examples=5)
    def test_various_input_sizes(self, w, h):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
        ])
        img = Image.new("RGB", (w, h), color=(0, 0, 255))
        result = transform(img)
        assert result.size == (512, 512)


class TestImageProcessing:
    def test_train_val_transforms_exist(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        ip = ImageProcessing()
        assert hasattr(ip, 'train_transforms')
        assert hasattr(ip, 'val_transforms')

    def test_val_has_no_random(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        from torchvision import transforms
        ip = ImageProcessing()
        val_str = str(ip.val_transforms)
        assert "RandomRotation" not in val_str
        assert "RandomHorizontalFlip" not in val_str
        assert "RandomVerticalFlip" not in val_str
        assert "ColorJitter" not in val_str


class TestOpenCVPreprocessingProperties:
    @given(st.integers(min_value=10, max_value=50))
    @settings(max_examples=5)
    def test_denoise_does_not_crash(self, h):
        img = Image.new("RGB", (h * 10, h * 10), color=(200, 100, 50))
        result = OpenCVPreprocessing()(img)
        assert result.size == (h * 10, h * 10)
