import clip
import torch
from torchvision import transforms

class SimplePrompt:
    input_type="image_embedding"
    """
    Simple image or text prompt that just compares the encoding of the image or text with image encoding.
    """
    @staticmethod
    def make_image_prompts(prompts, model_clip, clip_preprocessor, device, exclude=False):
        output = []

        for prompt in prompts:
            img = clip_preprocessor(prompt["prompt"]).unsqueeze(0).to(device)
            encoding = model_clip.encode_image(img)
            weight = torch.tensor(prompt["weight"], device=device)

            output.append(SimplePrompt(prompt["prompt"], encoding, prompt["weight"], exclude))

        return output

    @staticmethod
    def make_text_prompts(prompts, model_clip, device, exclude=False):
        output = []

        for prompt in prompts:
            t = clip.tokenize([prompt["prompt"]]).to(device)
            encoding = model_clip.encode_text(t).detach()
            weight = torch.tensor(prompt["weight"], device=device)

            output.append(SimplePrompt(prompt["prompt"], encoding, prompt["weight"], exclude))

        return output

    def __init__(self, prompt, encoding, weight, exclude=False):
        self.prompt = prompt
        self.encoding = encoding
        self.weight = weight
        self.exclude = exclude

    def loss(self, image_encoding):
        l = torch.cosine_similarity(image_encoding, self.encoding, -1).mean() * self.weight

        if not self.exclude:
            l *= -1.0

        self.last_loss = l

        return l

class ObjectiveCLIP:
    def __init__(self, prompts, agumenter, model_clip, device):
        self.prompts = prompts
        self.agumenter = agumenter
        self.model_clip = model_clip
        self.device = device

        self.norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711)).to(device)

    def __call__(self, img):
        img = self.agumenter(img)
        img = self.norm(img)

        img_encoding = self.model_clip.encode_image(img)

        loss = torch.tensor(0.0, device=self.device)
        for prompt in self.prompts:
            if prompt.input_type == "image_embedding":
                loss += prompt.loss(img_encoding)

        return loss
