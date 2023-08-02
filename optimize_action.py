import os
os.environ['IMAGEIO_FFMPEG_EXE']="/usr/bin/ffmpeg"
import torch
import moviepy.editor as mpy
from torch.autograd import Variable
import torchvision.transforms as transforms
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Meshes
from lavis.models import load_model_and_preprocess

from mvnet.models.render import DiffRender
from mvnet.models.skinning import LinearBlendSkinning
from mvnet.utils.transform import (
    rotation_6d_to_matrix,
    matrix_to_euler_angles
)
from mvnet.utils.utils import parse_bvh_to_frame


if __name__ == "__main__":
    device = torch.device("cuda")

    trg_file = "../MVNet/dataset/Amy/Baseball_Catcher_Pop_Fly.bvh"
    trg_fbx_path = os.path.join(*trg_file.split('/')[:-2])
    trg_list, trg_skeleton = parse_bvh_to_frame(trg_file, need_pose=False, fbx_path=trg_fbx_path, device=device)
    trg_skeleton = trg_skeleton.to(device)

    R, T = look_at_view_transform(dist=250,  device=device)
    render = DiffRender(R, T, image_size=224, sigma=1e-6, device=device, need_silhouette=False)
    lbs = LinearBlendSkinning()
    lbs.init(trg_skeleton, device)
    
    preprocess = transforms.Compose([
        # transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.Lambda(lambda x: transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(x).unsqueeze(0).to(device)),
    ])
    caption = ["Question: Describe the motion of the character in detail? Answer:"]

    ref_embedding = torch.load("./src_llm.pt")

    imgs = []
    ang = matrix_to_euler_angles(rotation_6d_to_matrix(torch.zeros_like(trg_list[0].ang)), 'XYZ').reshape(1, -1, 3).to(device)
    ang = Variable(ang.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([ang], lr=0.1, weight_decay=0.1)

    model, _, _ = load_model_and_preprocess(
        name="blip2_t5_instruct", model_type="flant5xxl", is_eval=True
    )

    for name, param in model.named_parameters():
        param.requires_grad = False
    
    for i in range(10000):
        optimizer.zero_grad()
        verts = lbs(ang)
        vert_pos = lbs.transform_to_pos(verts).squeeze()
        mesh = Meshes(vert_pos[None], trg_skeleton.faces[None], trg_skeleton.texture)
        images_rgb, _ = render(mesh)
        imgs.append((images_rgb.detach().cpu().numpy()*255.).astype('uint8'))
        images_rgb = preprocess(images_rgb.permute(2,0,1))
        features = model.forward_loss({"image":images_rgb, "text_input":caption}).encoder_last_hidden_state
        loss = torch.nn.functional.mse_loss(features, ref_embedding.to(features.device), reduction='sum')
        print("iter{} loss: {:4f}".format(i, loss.item()))
        loss.backward()
        optimizer.step()

        if i % 1000 == 0 and i > 0:
            clip = mpy.ImageSequenceClip(imgs, fps=10)
            clip.write_videofile("./optimization_{}.mp4".format(i))
    # get src image
