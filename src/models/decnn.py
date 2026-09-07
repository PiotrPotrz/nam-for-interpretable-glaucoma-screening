"""
Implementation code of Deep Explainable Concept-based Neural Network for Glaucoma Screening
"""
import os

import torch
import torch.nn as nn
import kornia

import segmentation_models_pytorch as smp


class NAM_MLP(nn.Module):
    def __init__(self,
                 network=[[1,2],[2,4],[4,6],[6,2],[2,1]], # [[1,2],[2,4],[4,8],[8,4],[4,2],[2,1]]
                 activation="relu"):
        super().__init__()
        if activation=="relu":
            self.activation = nn.ReLU()
        elif activation=="silu":
            self.activation = nn.SiLU()
        elif activation=="gelu":
            self.activation = nn.GELU()
        elif activation=="tanh":
            self.activation = nn.Tanh()
        elif activation=="elu":
            self.activation=nn.ELU()
        elif activation=="prelu":
            self.activation = nn.PReLU()


        layers = []
        for block in network:
            layers.append(self.__relu_linear(in_features=block[0], out_features=block[1]))

        self.sequence = nn.Sequential(*layers)

    def __relu_linear(self,in_features, out_features):
        if out_features !=1:
            block = nn.Sequential(
                nn.Linear(in_features, out_features),
                self.activation
            )
        else:
            block = nn.Linear(in_features, out_features)
        return block

    def forward(self, x):
        out = self.sequence(x)
        return out


class ConvColorDescriptor(nn.Module):
    """
    Mini CNN Network used for extraction of simple image concepts such as color, brightness, microstructure etc.
    It is shallow by design.
    """
    def __init__(self,activation="relu"):
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "prelu":
            self.activation = nn.PReLU()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        out = self.network(x)
        out = out.squeeze(-1)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_color_descriptors(image, mask):
        """
        Function calculates mean and standard deviation of all image channels ignoring places where mask is zero.
        image: (3, H, W)
        mask:  (H, W) bool
        """

        mask_exp = mask.unsqueeze(1)  # Bx1xHxW

        num_pixels = mask_exp.sum(dim=[2, 3]).clamp_min(1)

        mean = (image * mask_exp).sum(dim=[2, 3]) / num_pixels

        var = ((image - mean.unsqueeze(-1).unsqueeze(-1)) ** 2 * mask_exp).sum(dim=[2, 3]) / num_pixels
        std = torch.sqrt(var + 1e-8)

        return torch.cat([mean, std], dim=-1)

    @staticmethod
    def get_ISNT(H=512, W=512, device="cuda"):
        """
        The function gives masks used for ISNT sectors calculation.
        """
        cy, cx = H // 2, W // 2

        y, x = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="ij")

        angle = torch.atan2(cy - y, x - cx)  # range [-pi, pi]

        superior = (angle > torch.pi / 4) & (angle <= 3 * torch.pi / 4)
        inferior = (angle > -3 * torch.pi / 4) & (angle <= -torch.pi / 4)
        temporal = (angle > -torch.pi / 4) & (angle <= torch.pi / 4)
        nasal = (angle > 3 * torch.pi / 4) | (angle <= -3 * torch.pi / 4)

        superior = superior.float().to(device)
        inferior = inferior.float().to(device)
        temporal = temporal.float().to(device)
        nasal = nasal.float().to(device)
        return superior, inferior, temporal, nasal

    @staticmethod
    def reshape(image, mask, shape=(512,512)):
        image = image.unsqueeze(0)  # [1, C, H, W]
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


        image = torch.nn.functional.interpolate(
                image,  # [B, C, H, W]
                size=shape,
                mode="bilinear",
                align_corners=False
            )
        mask = torch.nn.functional.interpolate(
                mask.float(),  # [B, C, H, W]
                size=shape,
                mode="nearest",
            )

        # remove batch & channel dims
        image = image.squeeze(0)  # [C, H, W]
        mask = mask.squeeze(0).squeeze(0)  # [H, W]
        mask = mask.long()

        return image, mask

    def crop_to_disc(self, image, mask):
        """
        Function crops both image and mask to disc region.
        """
        bin_mask = (mask > 0)

        # bin_mask = bin_mask[0]

        images, masks = [], []

        for i in range(bin_mask.shape[0]):
            rows = torch.where(bin_mask[i].any(dim=1))[0]
            cols = torch.where(bin_mask[i].any(dim=0))[0]

            ymin, ymax = rows[0].item(), rows[-1].item()
            xmin, xmax = cols[0].item(), cols[-1].item()

            cropped = image[i, :, ymin:ymax + 1, xmin:xmax + 1]

            cropped_mask = mask[i,ymin:ymax + 1, xmin:xmax + 1]

            cropped, cropped_mask = self.reshape(cropped, cropped_mask)

            images.append(cropped)
            masks.append(cropped_mask)

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        return images, masks

    @staticmethod
    def get_boundary_region(image, mask):
        """
        Function creates a boundary region mask and crops it.
        """
        bin_mask = (mask > 0).to(image.device)
        kernel = torch.ones(12, 12).to(image.device)
        dilated_mask = kornia.morphology.dilation(bin_mask.unsqueeze(1), kernel, border_type='constant').to(image.device)
        boundary_mask = dilated_mask.squeeze(1).int() - bin_mask.int()
        boundary_image = image * boundary_mask.unsqueeze(1)
        return boundary_image, boundary_mask


    def get_disc_data(self, image, mask):
        """
        The function used to get information (colors and area) from ISNT like slices.
        """
        cr_im, cr_ma = self.crop_to_disc(image, mask)
        isnt_sectors = self.get_ISNT(cr_im.shape[-2], cr_im.shape[-1], image.device)

        # print(image.device)
        # print([isnt.device for isnt in isnt_sectors])

        # isnt_sectors = [isnt.to(image.device) for isnt in isnt_sectors]

        disc_mask = (cr_ma == 1)

        disc_sectors = [disc_mask * sector for sector in isnt_sectors]

        full_mask = cr_ma > 0
        disc_sectors_full = [full_mask * sector for sector in isnt_sectors]

        areas = [dm.sum(dim=[-2,-1]) / (fm.sum(dim=[-2,-1]) + 1e-12) for (dm, fm) in zip(disc_sectors, disc_sectors_full)]
        areas = torch.stack(areas).permute(1,0)
        colors = [self.get_color_descriptors(cr_im, sector) for sector in disc_sectors]

        colors = torch.stack(colors).permute(1,0,2)
        return areas, colors


    def get_boundary_data(self, image, mask):
        """
        Function used for extracting data from the borders of the optic disc.
        """
        bim, bma = self.get_boundary_region(image, mask)
        isnt_sectors = self.get_ISNT(bim.shape[-2], bim.shape[-1], image.device)
        disc_sectors = [bma * sector for sector in isnt_sectors]
        colors = [self.get_color_descriptors(bim, sector) for sector in disc_sectors]
        colors = torch.stack(colors).permute(1,0,2)
        return colors

    @staticmethod
    def get_enlongation(y_length, x_length):
        return x_length / (y_length + 1e-12)

    @staticmethod
    def get_circularity(mask):
        area = mask.sum(dim=[-2, -1])

        kernel = torch.ones(3, 3).to(mask.device)

        dilated_mask = kornia.morphology.dilation((mask > 0).unsqueeze(1), kernel, border_type='constant')
        eroded_mask = kornia.morphology.erosion((mask > 0).int().unsqueeze(1), kernel,
                                                border_type='constant')
        perimeter = (dilated_mask - eroded_mask).sum(dim=[-2, -1]) / 2

        circularity = 4 * torch.pi * area / (torch.pow(perimeter, 2) + 1e-12).squeeze(1)
        return circularity

    def get_geometry(self, mask):
        """
        Function used for extracting C-D ratios from CFI masks.
        """
        mask_cup = (mask == 2)
        mask_disc = (mask == 1)
        # area cdr
        num_cup = mask_cup.sum(dim=[-2, -1])
        num_disc = mask_disc.sum(dim=[-2, -1])
        area_cdr = num_cup / (num_disc + num_cup + 1e-12)

        # horizontal and vertical CDRs

        # extract bbox coordinates
        rows_cup = mask_cup.any(dim=2)  # (B,H)
        cols_cup = mask_cup.any(dim=1)  # (B,W)

        y_idx = torch.arange(rows_cup.shape[1], device=mask_cup.device)
        x_idx = torch.arange(cols_cup.shape[1], device=mask_cup.device)

        ymin_cup = torch.where(rows_cup, y_idx, rows_cup.shape[1]).min(dim=1).values
        ymax_cup = torch.where(rows_cup, y_idx, -1).max(dim=1).values

        xmin_cup = torch.where(cols_cup, x_idx, cols_cup.shape[1]).min(dim=1).values
        xmax_cup = torch.where(cols_cup, x_idx, -1).max(dim=1).values

        rows_disc = mask_disc.any(dim=2)  # (B,H)
        cols_disc = mask_disc.any(dim=1)  # (B,W)

        y_idx_disc = torch.arange(rows_disc.shape[1], device=mask_disc.device)
        x_idx_disc = torch.arange(cols_disc.shape[1], device=mask_disc.device)

        ymin_disc = torch.where(rows_disc, y_idx_disc, rows_disc.shape[1]).min(dim=1).values
        ymax_disc = torch.where(rows_disc, y_idx_disc, -1).max(dim=1).values

        xmin_disc = torch.where(cols_disc, x_idx_disc, cols_disc.shape[1]).min(dim=1).values
        xmax_disc = torch.where(cols_disc, x_idx_disc, -1).max(dim=1).values

        # get lengths
        y_cup = ymax_cup - ymin_cup
        x_cup = xmax_cup - xmin_cup

        y_disc = ymax_disc - ymin_disc
        x_disc = xmax_disc - xmin_disc

        # calculate ratios
        cdr_y = y_cup / (y_disc + 1e-12)
        cdr_x = x_cup / (x_disc + 1e-12)

        # calculate enlongation
        enl_cup = self.get_enlongation(y_cup, x_cup)
        enl_disc = self.get_enlongation(y_disc, x_disc)

        # calculate circularity
        circ_cup = self.get_circularity(mask_cup)
        circ_disc = self.get_circularity(mask_disc)

        return area_cdr, cdr_y, cdr_x, enl_cup, enl_disc, circ_cup, circ_disc

    @staticmethod
    def __normalize_per_image(image):
        """
        Function used for normalizing CFI images in such a way that they include mean color values in image
        and are [0,1].
        """
        image = image / image.mean(dim=[1, 2, 3], keepdim=True)
        maximum = torch.amax(image, dim=[1,2,3], keepdim=True)
        minimum = torch.amin(image, dim=[1,2,3], keepdim=True)
        normalized_image = (image - minimum) / (maximum - minimum)
        return normalized_image


    def forward(self, image, mask):
        # using RGB -> TO LAB conversion
        image = kornia.color.rgb_to_lab(image)

        image = self.__normalize_per_image(image)
        disc_areas, disc_colors = self.get_disc_data(image, mask)
        boundary_colors = self.get_boundary_data(image, mask)
        area_cdr, cdr_y, cdr_x, enl_cup, enl_disc, circ_cup, circ_disc = self.get_geometry(mask)
        colors_cup = self.get_color_descriptors(image, mask==2)

        vars_to_print = {
            "disc_areas": disc_areas,
            "disc_colors": disc_colors,
            "boundary_colors": boundary_colors,
            "area_cdr": area_cdr,
            "cdr_y": cdr_y,
            "cdr_x": cdr_x,
            "enl_cup": enl_cup,
            "enl_disc": enl_disc,
            "circ_cup": circ_cup,
            "circ_disc": circ_disc,
            "colors_cup": colors_cup,
        }

        return vars_to_print


# class FeatureExtractorWithCNN(FeatureExtractor):
#     def __init__(self):
#         super().__init__()
#
#     @staticmethod
#     def get_color_descriptors(image, mask):
#         """
#         Function crops relevant parts of the images, reshapes them and converts them to sequence.
#         """
#         mask = mask.bool()
#         # pixels = image * mask.unsqueeze(1)
#
#         bin_mask = mask[0]
#
#         rows = torch.where(bin_mask.any(dim=1))[0]
#         cols = torch.where(bin_mask.any(dim=0))[0]
#
#         ymin, ymax = rows[0].item(), rows[-1].item()
#         xmin, xmax = cols[0].item(), cols[-1].item()
#
#         cropped = image[:, :, ymin:ymax + 1, xmin:xmax + 1]
#
#         cropped = torch.functional.interpolate(
#             cropped,  # [B, C, H, W]
#             size=(50, 50),
#             mode="bilinear",
#             align_corners=False
#         )
#
#         seq = cropped.flatten(start_dim=-2)
#
#
#         return torch.cat([mean, std], dim=-1)


class DECNN(nn.Module):
    def __init__(self,
                 # network_one_feature=[[1,2],[2,4],[4,8],[8,4],[4,2],[2,1]],
                 # network_color = [[6,6],[6,4],[4,8],[8,4],[4,2],[2,1]],
                 network_one_feature=[[1,12],[12,16],[16,16],[16,8],[8,4],[4,1]],
                 network_color = [[6,12],[12,16],[16,16],[16,8],[8,4],[4,1]],
                 activation="relu"
                 ):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.area_nets = nn.ModuleDict({
            f"area_{i}" :NAM_MLP(network=network_one_feature, activation=activation)
            for i in range(4)
        })
        self.cdr_nets = nn.ModuleDict({
            f"cdr_{i}": NAM_MLP(network=network_one_feature, activation=activation)
            for i in range(3)
        })
        self.color_nets = nn.ModuleDict({
            f"color_{i}": NAM_MLP(network=network_color, activation=activation) # [[6,6],[6,4],[4,8],[8,4],[4,2],[2,1]]
            for i in range(9)
        })
        self.shape_nets = nn.ModuleDict({
            f"shape_{i}": NAM_MLP(network=network_one_feature, activation=activation)
            for i in range(4)
        })

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, image, seg_mask):
        features = self.feature_extractor(image, seg_mask)


        area_outs = []

        x = features["disc_areas"]  # [B, 4]

        B = x.shape[0]

        for i in range(x.shape[1]):
            xi = x[:, i:i + 1]  # [B, 1]
            fi = self.area_nets[f"area_{i}"](xi)
            fi = fi.view(fi.shape[0], 1)
            area_outs.append(fi)


        cdr_outs = []

        x = [features["area_cdr"], features["cdr_y"], features["cdr_x"]]

        for i in range(len(x)):
            xi = x[i]
            fi = self.cdr_nets[f"cdr_{i}"](xi.unsqueeze(1))
            fi = fi.view(fi.shape[0], 1)
            cdr_outs.append(fi)

        color_outs = []

        disc_colors = features["disc_colors"]  # [B, 4, 6]
        boundary_colors = features["boundary_colors"]  # [B, 4, 6]
        colors_cup = features["colors_cup"].view(B, 1, 6)
        x = torch.cat([disc_colors, boundary_colors, colors_cup], dim=1)

        for i in range(x.shape[1]):
            xi = x[:, i:i + 1,:]
            fi = self.color_nets[f"color_{i}"](xi)
            fi = fi.view(fi.shape[0], 1)
            color_outs.append(fi)

        shape_outs = []

        x = [features["enl_cup"], features["enl_disc"], features["circ_cup"], features["circ_disc"]]

        for i in range(len(x)):
            xi = x[i]
            fi = self.shape_nets[f"shape_{i}"](xi.unsqueeze(1))
            fi = fi.view(fi.shape[0], 1)
            shape_outs.append(fi)

        # calculating NAM output
        full_outs = area_outs + cdr_outs + color_outs + shape_outs
        out = 0
        for o in full_outs:
            out += o
        out+= self.bias
        return out, full_outs


class DECNN_WITH_SEGMENTOR(nn.Module):
    def __init__(self,
                 network_one_feature=[[1, 2], [2, 4], [4, 8], [8, 4], [4, 2], [2, 1]], # [[1, 2], [2, 4], [4, 8], [8, 4], [4, 2], [2, 1]]
                 network_color=[[6, 6], [6, 4], [4, 8], [8, 4], [4, 2], [2, 1]], # [[6, 6], [6, 4], [4, 8], [8, 4], [4, 2], [2, 1]]
                 activation = "relu",
                 dataset="RIM_ONE_DL_RANDOM"
                 ):
        super().__init__()
        self.decnn = DECNN(network_one_feature, network_color, activation=activation)
        self.segmentor = smp.create_model(arch="UNetPlusPlus", encoder_name="tu-hrnet_w48_ssld", encoder_weights=None, in_channels=3, classes=3)
        # if athena:
        if dataset == "RIM_ONE_DL_RANDOM":
            if "plgrid" in os.getcwd():
                self.segmentor.load_state_dict(torch.load("/net/tscratch/people/plgppotrz/GLAUCOMA/losses_test/SEG(0.2, 0.8, 1.5)$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$RIM_ONE_DL_RANDOM$_best_iou.pth"))
            else:
                self.segmentor.load_state_dict(torch.load("/home/user/python_projects/glaucoma/GLAUCOMA/SEG(0.2, 0.8, 1.5)$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$RIM_ONE_DL_RANDOM$_best_iou.pth"))
        elif dataset == "RIM_ONE_DL":
            if "plgrid" in os.getcwd():
                self.segmentor.load_state_dict(torch.load("/net/tscratch/people/plgppotrz/GLAUCOMA/HOSPITAL_SEGMENTOR/SEG2026_02_24 18_15_39$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$RIM_ONE_DL$_best_iou.pth"))
            else:
                self.segmentor.load_state_dict(torch.load("SEG2026_02_24 18_15_39$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$RIM_ONE_DL$_best_iou.pth"))

        elif dataset == "REFUGE":
            if "plgrid" in os.getcwd():
                self.segmentor.load_state_dict(torch.load(
                    "/net/tscratch/people/plgppotrz/GLAUCOMA/REFUGE2/SEG2026_02_17 11_20_13$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$REFUGE$_best_iou.pth"))
            else:
                self.segmentor.load_state_dict(torch.load(
                    "/home/user/python_projects/glaucoma/GLAUCOMA/SEG2026_02_17 11_20_13$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$REFUGE$_best_iou.pth"))
        else:
            raise ValueError("Unknown dataset")
        for p in self.segmentor.parameters():
            p.requires_grad = False
        self.segmentor.eval()

    def forward(self, image):
        # calculating segmentation masks
        self.segmentor.eval()
        with torch.no_grad():
            segmentation_mask = self.segmentor(image)
        segmentation_mask = torch.argmax(segmentation_mask, dim=1)

        # performing classification
        pred, parts = self.decnn(image, segmentation_mask)
        return pred, parts


class FEATURE_EXTRACTOR_WITH_SEGMENTOR(nn.Module):
    def __init__(self,
                 dataset="RIM_ONE_DL_RANDOM"
                 ):
        super().__init__()
        self.segmentor = smp.create_model(arch="UNetPlusPlus", encoder_name="tu-hrnet_w48_ssld", encoder_weights=None, in_channels=3, classes=3)
        self.extractor = FeatureExtractor()
        # if athena:
        if dataset == "RIM_ONE_DL_RANDOM":
            if "plgrid" in os.getcwd():
                self.segmentor.load_state_dict(torch.load("/net/tscratch/people/plgppotrz/GLAUCOMA/losses_test/SEG(0.2, 0.8, 1.5)$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$RIM_ONE_DL_RANDOM$_best_iou.pth"))
            else:
                self.segmentor.load_state_dict(torch.load("/home/user/python_projects/glaucoma/GLAUCOMA/SEG(0.2, 0.8, 1.5)$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$RIM_ONE_DL_RANDOM$_best_iou.pth"))
        elif dataset == "RIM_ONE_DL":
            if "plgrid" in os.getcwd():
                self.segmentor.load_state_dict(torch.load("/net/tscratch/people/plgppotrz/GLAUCOMA/HOSPITAL_SEGMENTOR/SEG2026_02_24 18_15_39$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$RIM_ONE_DL$_best_iou.pth"))
            else:
                self.segmentor.load_state_dict(torch.load("SEG2026_02_24 18_15_39$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$RIM_ONE_DL$_best_iou.pth"))

        elif dataset == "REFUGE":
            if "plgrid" in os.getcwd():
                self.segmentor.load_state_dict(torch.load(
                    "/net/tscratch/people/plgppotrz/GLAUCOMA/REFUGE2/SEG2026_02_17 11_20_13$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$REFUGE$_best_iou.pth"))
            else:
                self.segmentor.load_state_dict(torch.load(
                    "/home/user/python_projects/glaucoma/GLAUCOMA/SEG2026_02_17 11_20_13$UnetPlusPlus$tu-hrnet_w48_ssld$ce$adam$100$a2$cosine$0.0001$True$8$REFUGE$_best_iou.pth"))
        else:
            raise ValueError("Unknown dataset")
        for p in self.segmentor.parameters():
            p.requires_grad = False
        self.segmentor.eval()

    @staticmethod
    def get_feature_names():
        sectors = ["superior", "inferior", "temporal", "nasal"]
        color_stats = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std"]

        names = []

        names += [f"disc_area_{s}" for s in sectors]

        for region in ["disc", "boundary"]:
            for sector in sectors:
                for stat in color_stats:
                    names.append(f"{region}_{sector}_{stat}")

        names += [
            "area_cdr",
            "cdr_y",
            "cdr_x",
            "enl_cup",
            "enl_disc",
            "circ_cup",
            "circ_disc",
        ]

        names += [f"cup_{stat}" for stat in color_stats]

        return names

    @staticmethod
    def flatten_features(features):
        """
        Converts FeatureExtractor output dict into flat tensor [B, n_features].
        """

        flat = [
            features["disc_areas"],  # [B, 4]
            features["disc_colors"].flatten(start_dim=1),  # [B, 4*6]
            features["boundary_colors"].flatten(start_dim=1),  # [B, 4*6]

            features["area_cdr"].unsqueeze(1),
            features["cdr_y"].unsqueeze(1),
            features["cdr_x"].unsqueeze(1),
            features["enl_cup"].unsqueeze(1),
            features["enl_disc"].unsqueeze(1),
            features["circ_cup"].unsqueeze(1),
            features["circ_disc"].unsqueeze(1),

            features["colors_cup"],  # [B, 6]
        ]

        return torch.cat(flat, dim=1)

    def forward(self, image):
        self.segmentor.eval()

        with torch.no_grad():
            segmentation_mask = self.segmentor(image)

        segmentation_mask = torch.argmax(segmentation_mask, dim=1)

        features = self.extractor(image, segmentation_mask)
        features = self.flatten_features(features)

        # if self.return_names:
        #     return features, self.get_feature_names()

        return features



if __name__ == "__main__":
    extractor = FeatureExtractor()

    from src.dataset import RimOneDataset
    from torch.utils.data import DataLoader
    dataset = RimOneDataset(dataset="RIM_ONE_DL", mode="train", include_segmentation=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    extractor = FEATURE_EXTRACTOR_WITH_SEGMENTOR(dataset="RIM_ONE_DL")

    for i, l in loader:
        features = extractor(i)
        break
    print(features.shape)