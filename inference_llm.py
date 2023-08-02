import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from lavis.models import load_model_and_preprocess

if __name__ == "__main__":
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct", model_type="flant5xxl", is_eval=True
    )
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.Lambda(lambda x: x.resize((224, 224))),
        transforms.Lambda(lambda x: vis_processors["eval"](x).unsqueeze(0).to(device)),
    ])

    img = np.load("src_img.npy")[32]
    img = preprocess(img)

    caption = ["Question: Describe the motion of the character in detail? Answer:"]
    result = model.forward_loss({"image":img, "text_input":caption})
    torch.save(result.encoder_last_hidden_state.detach().cpu(), "src_llm.pt")