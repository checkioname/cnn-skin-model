import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from domain.SetupModel import SetupModel


class TestModelForward:
    device = torch.device("cpu")

    def _test_model_output(self, model_name):
        setup = SetupModel(model_name)
        model, loss_fn, optimizer, scheduler = setup.setup_model(self.device)
        model.eval()

        dummy = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = model(dummy)

        assert out.shape == (2, 1), f"Esperado (2,1), got {out.shape}"
        assert out.dtype == torch.float32
        assert not torch.isnan(out).any(), "NaN na saida do modelo"

    def test_vgg16(self):
        self._test_model_output("vgg16")

    def test_resnet152(self):
        self._test_model_output("resnet152")

    def test_vit(self):
        self._test_model_output("vit")

    def test_swin(self):
        self._test_model_output("swin")


class TestSetupModel:
    def test_invalid_model_raises(self):
        setup = SetupModel("invalid_model")
        try:
            setup.setup_model(torch.device("cpu"))
            assert False, "Deveria levantar ValueError"
        except ValueError:
            pass

    def test_returns_tuple(self):
        setup = SetupModel("vgg16")
        result = setup.setup_model(torch.device("cpu"))
        assert len(result) == 4
        model, loss_fn, optimizer, scheduler = result
        assert model is not None
        assert loss_fn is not None
        assert optimizer is not None
        assert scheduler is not None
