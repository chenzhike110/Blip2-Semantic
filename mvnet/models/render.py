import torch
import numpy as np
import torch.nn as nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    MeshRenderer,
    SoftPhongShader,
    SoftSilhouetteShader,
    MeshRasterizer,
    BlendParams,
    PointsRasterizationSettings,
    PointLights,
    RasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizer
)

class DiffRender(nn.Module):
    """
    Differential Render 
    """
    def __init__(self, R, T, zfar=400, sigma=1e-4, image_size=512, device=torch.device('cpu'), need_silhouette=True) -> None:
        super().__init__()
        self.point_size = image_size // 4
        self.device = device
        self.lights = PointLights(device=device, location=[[0.0, 200.0, 0.0]])
        self.need_silhouette = need_silhouette
        self.renderers = []
        self.cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                           T=T[None, i, ...], zfar=zfar) for i in range(R.shape[0])]
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            # blur_radius=0,
            faces_per_pixel=15, 
            # perspective_correct=False, 
        )
        self.point_raster_settings = PointsRasterizationSettings(
            image_size=self.point_size, 
            radius = 0.01,
            points_per_pixel = 1
        )

        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras[0], 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader()
        )

        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras[0], 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=self.cameras[0],
                lights=self.lights,
                # blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))
            )
        )
        self.rasterizer = PointsRasterizer(cameras=self.cameras[0], raster_settings=self.point_raster_settings)
        self.renderer_point = PointsRenderer(
            rasterizer=self.rasterizer,
            compositor=AlphaCompositor()
        )

    def forward_point(self, verts):
        point_cloud = Pointclouds(points=[verts])
        fragments = self.rasterizer(point_cloud).idx
        cluster_centers = torch.ones((verts.shape[0], 2)) * -1.0
        for i in range(verts.shape[0]):
            pixel = (fragments == i).nonzero()[:, 1:3].float()
            cluster_centers[i] = torch.mean(pixel, dim=0)
        
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        # plt.imshow((images[0, ..., :3]).cpu().numpy())
        # plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], marker='*', s=10, edgecolor='white', linewidth=1.25)
        # plt.axis("off")
        # plt.savefig("./test.jpg")
        # plt.show()
        cluster_centers = cluster_centers.detach().cpu().numpy()
        return cluster_centers / self.point_size

    def forward_silhouette(self, mesh, cameras):
        images_silhouette = []
        for camera in cameras:
            images_silhouette.append(self.renderer_silhouette(mesh, cameras=camera, lights=self.lights)[..., 3])
        images_silhouette = torch.stack(images_silhouette, dim=1).squeeze()
        return images_silhouette

    def forward_rgb(self, mesh, cameras):
        images_rgb = []
        for camera in cameras:
            images_rgb.append(self.renderer_textured(mesh, cameras=camera, lights=self.lights)[..., :3])
        images_rgb = torch.stack(images_rgb, dim=1).squeeze()
        return images_rgb

    def forward(self, mesh, camera_id="all"):
        if camera_id == "all":
            cameras = self.cameras
        else:
            cameras = [self.cameras[i] for i in camera_id]        
        images_rgb = self.forward_rgb(mesh, cameras)
        if self.need_silhouette:
            images_silhouette = self.forward_silhouette(mesh, cameras)
        else:
            images_silhouette = None

        return images_rgb, images_silhouette