import ray
import torch


class Actor:
    def __init__(self, model):
        self.model = ray.get(model)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def __call__(self, batch):
        with torch.inference_mode():
            output = self.model(torch.as_tensor(batch["image"], device=self.device))
            return {"predictions": output.cpu().numpy()}
