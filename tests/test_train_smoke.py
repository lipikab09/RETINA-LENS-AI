from retina_lens.config import load_config
from retina_lens.train import main


def test_two_epoch_cpu_smoke(tmp_path):
    cfg = load_config("configs/debug_cpu.yaml", {"paths": {"runs": str(tmp_path)}})
    run = main(cfg); assert (run / "best.pt").exists() and (run / "history.json").exists()
    from retina_lens.inference import Predictor
    import numpy as np
    r = Predictor(run / "best.pt", "cpu").predict(np.random.randint(30, 200, (96, 96, 3), dtype=np.uint8), mc=True, cam=True)
    assert abs(sum(r["probabilities"]) - 1) < 1e-4 and r["synthetic_model"] is True
