import clip
import torch
from torchvision import transforms
from tabulate import tabulate

class SimplePrompt:
    """
    Simple image or text prompt that just compares the encoding of the image or text with image encoding.
    """
    @staticmethod
    def make_from_image(filepath, weight, model_clip, clip_preprocessor, device, exclude=False):
        img = clip_preprocessor(filepath).unsqueeze(0).to(device)
        encoding = model_clip.encode_image(img)
        weight = torch.tensor(weight, device=device)

        return SimplePrompt(filepath, weight, encoding, exclude)

    @staticmethod
    def make_from_text(text, weight, model_clip, device, exclude=False):
        t = clip.tokenize([text]).to(device)
        encoding = model_clip.encode_text(t).detach()
        weight = torch.tensor(weight, device=device)

        return SimplePrompt(text, weight, encoding, exclude)

    def __init__(self, prompt, weight, encoding, exclude=False):
        self.prompt = prompt
        self.weight = weight
        self.encoding = encoding
        self.exclude = exclude

    def __call__(self, image_encoding):
        l = torch.cosine_similarity(image_encoding, self.encoding, -1).mean() * self.weight

        if not self.exclude:
            l *= -1.0

        self.last_loss = l

        return l

class ObjectiveCLIP:
    """
    Objective that uses CLIP image encoder.
    """
    def __init__(self, prompts, agumenter, model_clip, device):
        self.prompts = prompts
        self.agumenter = agumenter
        self.model_clip = model_clip
        self.device = device

        self.norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711)).to(device)

    def __call__(self, img):
        img_agu = self.agumenter(img)
        img_encoding = self.model_clip.encode_image(self.norm(img_agu))

        loss = torch.tensor(0.0, device=self.device)
        for prompt in self.prompts:
            if isinstance(prompt, SimplePrompt):
                loss += prompt(img_encoding)
            else:
                loss += prompt(img)

        return loss

    def print_objective_loss_report(self):
        """
        Print detailed breakdown of what is the loss value for all prompts.
        """
        table = [["Prompt", "Exclude", "Final loss", "Unweighted final loss"]]
        for prompt in self.prompts:
            table.append([prompt.prompt, prompt.exclude, float(prompt.last_loss),
                          float(prompt.last_loss) * (1/float(prompt.weight))])

        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
