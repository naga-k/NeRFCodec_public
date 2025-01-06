import pdb

import torch

from .tensorBase import *
from .qat import qfn, qfn2

from compressai.ops import compute_padding
from compressai.models.google import ScaleHyperprior

class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, device, **kargs)
        

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):

        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach()
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return sigma_feature, app_features

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])
    
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        # plane_coef[0] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[0].data, size=(res_target[1], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[0] = torch.nn.Parameter(
        #     F.interpolate(line_coef[0].data, size=(res_target[2], 1), mode='bilinear', align_corners=True))
        # plane_coef[1] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[1].data, size=(res_target[2], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[1] = torch.nn.Parameter(
        #     F.interpolate(line_coef[1].data, size=(res_target[1], 1), mode='bilinear', align_corners=True))
        # plane_coef[2] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[2].data, size=(res_target[2], res_target[1]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[2] = torch.nn.Parameter(
        #     F.interpolate(line_coef[2].data, size=(res_target[0], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        scale = res_target[0]/self.line_coef.shape[2] #assuming xyz have the same scale
        plane_coef = F.interpolate(self.plane_coef.detach().data, scale_factor=scale, mode='bilinear',align_corners=True)
        line_coef  = F.interpolate(self.line_coef.detach().data, size=(res_target[0],1), mode='bilinear',align_corners=True)
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.compute_stepSize(res_target)
        print(f'upsamping to {res_target}')


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        self.compression = False
        # self.return_likelihoods = kargs.get('rate_penalty', False)
        self.compression_strategy = kargs.get('compression_strategy', None)
        self.compress_before_volrend = kargs.get('compress_before_volrend', False)
        self.mode = "train"
        self.using_external_codec = False
        self.additional_vec = False
        self.vec_qat = kargs.get('vec_qat', False)
        self.decode_from_latent_code = kargs.get('decode_from_latent_code', False)

    def init_image_codec(self):
        import torch.nn as nn
        from compressai.zoo import image_models as pretrained_models
        def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
            # TODO: to check .eval() ?
            return pretrained_models[model](
                quality=quality, metric=metric, pretrained=True, progress=False
            )

        self.image_codec = load_pretrained("bmshj2018-hyperprior", "mse", 5).to(self.device)
        print("Eval :", not self.image_codec.training, "Train:" , self.image_codec.training)
        self.compression = True

    def get_optparam_from_image_codec(self, lr=1e-4, fix_encoder=False):
        if fix_encoder:
            pdb.set_trace()
            grad_vars = [{'params': self.image_codec.h_a.parameters(), 'lr': 0.},
                         {'params': self.image_codec.g_a.parameters(), 'lr': 0.},
                         {'params': self.image_codec.h_s.parameters(), 'lr': lr},
                         {'params': self.image_codec.g_s.parameters(), 'lr': lr},
                         {'params': self.image_codec.entropy_bottleneck.parameters(), 'lr': lr},
                         {'params': self.image_codec.gaussian_conditional.parameters(), 'lr': lr},]
        else:
            grad_vars = [{'params': self.image_codec.parameters(), 'lr': lr}]
        return grad_vars

    def init_feat_codec(self, codec_ckpt_path='', loading_pretrain_param=True, adaptor_q_bit=8, codec_backbone_type="cheng2020-anchor"):
        from .imageCoder import AdaptorScaleHyperprior, AdaptorMeanScaleHyperprior, AdaptorCheng2020Anchor, AdaptorCheng2020Attention
        from compressai.zoo.image import model_urls, cfgs
        from compressai.zoo.pretrained import load_pretrained
        from torch.hub import load_state_dict_from_url

        feat_codec_dict = {
            "bmshj2018-hyperprior": AdaptorScaleHyperprior,
            "mbt2018-mean": AdaptorMeanScaleHyperprior,
            "cheng2020-anchor": AdaptorCheng2020Anchor,
            "cheng2020-attn": AdaptorCheng2020Attention
        }

        architecture = codec_backbone_type #"bmshj2018-hyperprior"
        metric = "mse"
        # quality = 8 # balle2017, 1-8, 5
        quality = 6 # cheng2020, 1-6

        if self.decode_from_latent_code:
            feat_codec = OnlyDecoderCheng2020Anchor
        else:
            feat_codec = feat_codec_dict[architecture]

        if codec_ckpt_path == "":
            url = model_urls[architecture][metric][quality]
            state_dict = load_state_dict_from_url(url, progress=True)
            state_dict = load_pretrained(state_dict)
        else:
            codec_ckpt = torch.load(codec_ckpt_path)

        self.latent_code_ch = cfgs[architecture][quality][-1]
        # ch of one plane (from density triplane) : 16

        self.den_feat_codec = feat_codec(self.density_n_comp[0], *cfgs[architecture][quality], q_bit=adaptor_q_bit)
        if loading_pretrain_param:
            if codec_ckpt_path == "":
                missing_keys, unexpected_keys = self.den_feat_codec.load_state_dict(state_dict, strict=False)
                self.den_feat_codec.reload_from_pretrained() # 也会load prob. density function 的先验
            else:
                self.den_feat_codec.load_state_dict(codec_ckpt["den_feat_codec"])
        self.den_feat_codec.to(self.device) # .eval()

        # ch of one plane (from appearance triplane) : 48

        self.app_feat_codec = feat_codec(self.app_n_comp[0], *cfgs[architecture][quality], q_bit=adaptor_q_bit)
        if loading_pretrain_param:
            if codec_ckpt_path == "":
                missing_keys, unexpected_keys = self.app_feat_codec.load_state_dict(state_dict, strict=False)
                self.app_feat_codec.reload_from_pretrained()
            else:
                self.app_feat_codec.load_state_dict(codec_ckpt["app_feat_codec"])
        # pdb.set_trace()
        self.app_feat_codec.to(self.device)

        self.compression = True

        if self.decode_from_latent_code:
            self.latent_z_size = torch.ceil(self.gridSize / 64.0) # z size
            self.latent_y_size = self.latent_z_size * 4
            self.den_latent_y, self.den_latent_z = self.init_one_latent_code(self.latent_y_size, self.latent_z_size, 0.1, self.device)
            self.app_latent_y, self.app_latent_z = self.init_one_latent_code(self.latent_y_size, self.latent_z_size, 0.1, self.device)


    def get_optparam_from_feat_codec(self, lr_transform=1e-4, fix_decoder_prior=False, fix_encoder_prior=False):
        aux_grad_vars = None
        # --------------------------- type 1 ----------------------------------- #

        params_dict = dict(self.named_parameters())

        adaptor_param_list = {
            name
            for name, param in self.named_parameters()
            if param.requires_grad and 'adaptor' in name
        }
        prior_param_list = {
            name
            for name, param in self.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles") and 'adaptor' not in name and 'codec' in name
        }
        aux_param_list = {
            name
            for name, param in self.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        }

        if fix_decoder_prior:
            prior_param_to_be_excluded = [name for name in prior_param_list if 'g_s' in name or 'h_s' in name]
            prior_param_list = [name for name in prior_param_list if name not in prior_param_to_be_excluded]

        if fix_encoder_prior:
            lr_enc_prior = 0
            print("priors in enc. is fixed")
        else:
            print("priors in enc. is not fixed")
            lr_enc_prior = 1e-5

        grad_vars = [
            {'params': (params_dict[name] for name in sorted(adaptor_param_list)), 'lr': lr_transform},
            {'params': (params_dict[name] for name in sorted(prior_param_list)),   'lr': lr_enc_prior}, # 0 -> 1e-5? 2e-4
        ]

        aux_grad_vars = [
            {'params': (params_dict[name] for name in sorted(aux_param_list)), 'lr': 1e-3},
        ]


        return grad_vars, aux_grad_vars

    def load_img_codec_ckpt(self, ckpt_dir, ckpt=None):
        if ckpt is not None:
            img_codec_state_dict = torch.load(ckpt)
        else:
            img_codec_state_dict = torch.load(ckpt_dir + f'/img_codec.ckpt')
        self.image_codec.load_state_dict(img_codec_state_dict)

    def save_img_codec_ckpt(self, ckpt_dir):
        torch.save(self.image_codec.state_dict(), ckpt_dir + f'/img_codec.ckpt')

    def load_feat_codec_ckpt(self, ckpt_dir):
        app_codec_state_dict = torch.load(ckpt_dir + f'/app_codec.ckpt')
        self.app_feat_codec.load_state_dict(app_codec_state_dict)

        den_codec_state_dict = torch.load(ckpt_dir + f'/den_codec.ckpt')
        self.den_feat_codec.load_state_dict(den_codec_state_dict)

    def save_feat_codec_ckpt(self, ckpt_dir):
        # pass
        torch.save(self.app_feat_codec.state_dict(), ckpt_dir + f'/app_codec.ckpt')
        torch.save(self.den_feat_codec.state_dict(), ckpt_dir + f'/den_codec.ckpt')


    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def init_one_latent_code(self, y_gridSize, z_gridSize, scale, device):
        latent_code_y, latent_code_z = [], []
        # gridSize = [20, 20, 20]
        for i in range(len(self.vecMode)):
            mat_id_0, mat_id_1 = self.matMode[i]
            latent_code_y.append(torch.nn.Parameter(
                scale * torch.randn((1, self.latent_code_ch, int(y_gridSize[mat_id_1]), int(y_gridSize[mat_id_0])))))
            latent_code_z.append(torch.nn.Parameter(
                scale * torch.randn((1, self.latent_code_ch, int(z_gridSize[mat_id_1]), int(z_gridSize[mat_id_0])))))  #

        return torch.nn.ParameterList(latent_code_y).to(device), torch.nn.ParameterList(latent_code_z).to(device)

    def init_additional_volume(self, device):
        self.additional_vec = True
        self.additional_density_line = self.init_additional_svd(self.density_n_comp, self.gridSize*2, 0.1, device)
        self.additional_app_line = self.init_additional_svd(self.app_n_comp, self.gridSize*2, 0.1, device)

    def init_additional_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[i], 1))))

        return torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001, fix_plane=False):
        grad_vars = [{'params': self.basis_mat.parameters(), 'lr':lr_init_network}]

        lr_init_spatial_plane = 0 if fix_plane else lr_init_spatialxyz

        grad_vars += [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                      {'params': self.density_plane, 'lr': lr_init_spatial_plane},
                      {'params': self.app_line, 'lr': lr_init_spatialxyz},
                      {'params': self.app_plane, 'lr': lr_init_spatial_plane}]

        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def get_additional_optparam_groups(self, lr_init_spatialxyz = 0.02,):
        grad_vars = [{'params': self.additional_density_line, 'lr':lr_init_spatialxyz},
                     {'params': self.additional_app_line, 'lr':lr_init_spatialxyz}]
        return grad_vars

    def get_latent_code_groups(self, lr_latent_code = 0.002):
        grad_vars = [{'params': self.den_latent_y, 'lr': lr_latent_code},
                     {'params': self.den_latent_z, 'lr': lr_latent_code},
                     {'params': self.app_latent_y, 'lr': lr_latent_code},
                     {'params': self.app_latent_z, 'lr': lr_latent_code}]
        return grad_vars

    def batchwise_feature_compression(self, plane, return_likelihoods=True):

        # normalize(channel-wise min max -> 1,H,W)
        # min_vec, _ = torch.min(plane, dim=1)
        # max_vec, _ = torch.max(plane, dim=1)

        # normalize(plane-wise min max -> C,1,1)
        min_vec, _ = torch.min(plane.view(*plane.shape[:2], -1), dim=-1)
        max_vec, _ = torch.max(plane.view(*plane.shape[:2], -1), dim=-1)
        min_vec, max_vec = min_vec.view([*min_vec.shape, 1, 1]), max_vec.view([*max_vec.shape, 1, 1])

        norm_plane = (plane - min_vec) / (max_vec - min_vec)

        # spatial padding
        h, w = norm_plane.size(2), norm_plane.size(3)
        pad, unpad = compute_padding(h, w, min_div=2 ** 6)  # pad to allow 6 strides of 2
        norm_plane_padded = F.pad(norm_plane, pad, mode="constant", value=0)

        # channel padding
        pad_h, pad_w = norm_plane_padded.size(2), norm_plane_padded.size(3)
        channel_pad_flag = norm_plane_padded.size(1) % 3 > 0
        if channel_pad_flag:
            num_ch_pad = 3 - norm_plane_padded.size(1) % 3
            norm_plane_padded = torch.cat([norm_plane_padded, torch.zeros([1, num_ch_pad, pad_h, pad_w], device=norm_plane_padded.device)], dim=1)

        # split channel into batches with 3 channels
        pad_ch = norm_plane_padded.size(1)
        x = norm_plane_padded.reshape([pad_ch//3, 3, pad_h, pad_w])

        # network inference(batch wise)
        ### TODO: change feed forward to adapt to train and eval mode
        out_net = self.image_codec.forward(x)

        # get rid of padding
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_net["x_hat"] = out_net["x_hat"].reshape([1, -1, h, w])
        if channel_pad_flag:
            out_net["x_hat"] = out_net["x_hat"][:, 0:out_net["x_hat"].size(1) - num_ch_pad, ]

        # de-norm
        rec_plane = out_net["x_hat"] * (max_vec - min_vec) + min_vec

        if return_likelihoods:
            return rec_plane, out_net # shape: B, 192, latent_h, latent_w
        else:
            return rec_plane, None

    def feature_compression_via_feat_coder(self, plane, feat_codec, return_likelihoods=True, mode="train"):
        # TODO: to normalize or not to normalize? maybe need exp.
        # normalize(plane-wise min max -> C,1,1)
        min_vec, _ = torch.min(plane.view(*plane.shape[:2], -1), dim=-1)
        max_vec, _ = torch.max(plane.view(*plane.shape[:2], -1), dim=-1)
        min_vec, max_vec = min_vec.view([*min_vec.shape, 1, 1]), max_vec.view([*max_vec.shape, 1, 1])

        norm_plane = (plane - min_vec) / (max_vec - min_vec)

        # spatial padding
        h, w = norm_plane.size(2), norm_plane.size(3)
        # pad to allow 6 strides of 2
        pad, unpad = compute_padding(h, w, min_div=2 ** 6)
        norm_plane_padded = F.pad(norm_plane, pad, mode="constant", value=0)

        # feed forward
        ### TODO: change feed forward to adapt to train and eval mode
        x = norm_plane_padded
        if mode == "train":
            out_net = feat_codec.forward(x)
        elif mode == "eval":
            out_enc = feat_codec.compress(x)
            out_dec = feat_codec.decompress(out_enc["strings"], out_enc["shape"])
            # pdb.set_trace()
            out_net = out_dec
            out_net.update({"strings_length": sum(len(s[0]) for s in out_enc["strings"])})
            # pass

        # get rid of padding
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_net["x_hat"] = out_net["x_hat"].reshape([1, -1, h, w])

        # de-norm
        rec_plane = out_net["x_hat"] * (max_vec - min_vec) + min_vec

        if return_likelihoods:
            return rec_plane, out_net # shape: [????]
        else:
            return rec_plane, None

    def get_rate(self):
        return self.density_likelihood_list, self.app_likelihood_list

    def get_aux_loss(self):
        if self.compression_strategy == "batchwise_img_coding":
            pass
        elif self.compression_strategy == "adaptor_feat_coding":
            den_aux_loss = self.den_feat_codec.aux_loss()
            app_aux_loss = self.app_feat_codec.aux_loss()

            aux_loss = den_aux_loss + app_aux_loss
        # pdb.set_trace()
        return aux_loss

    def compute_feat_diff(self):
        self.den_diff = []
        for idx, ref_item in enumerate(self.ref_den_feat_list):
            # pdb.set_trace()
            den_diff = torch.mean( torch.abs(self.rec_density_feat_list[idx] - ref_item) )
            self.den_diff.append(den_diff)

        self.app_diff = []
        for idx, ref_item in enumerate(self.ref_app_feat_list):
            app_diff = torch.mean( torch.abs(self.rec_app_feat_list[idx] - ref_item) )
            self.app_diff.append(app_diff)

        return self.den_diff, self.app_diff

    def set_external_codec_flag(self):
        self.compression = True
        self.using_external_codec = True

    ### for generalizable training
    def compress_with_external_codec(self, den_feat_codec, app_feat_codec, mode="train"): # mode = "train"/"eval"
        self.map_fn = torch.nn.Tanh()
        self.den_rec_plane = []
        self.den_likelihood = []
        for idx_plane in range(len(self.density_plane)):
            rec_plane, likelihood = \
                self.feature_compression_via_feat_coder(self.map_fn(self.density_plane[idx_plane]),
                                                        den_feat_codec,
                                                        mode=mode)
            # pdb.set_trace()
            self.den_rec_plane.append(rec_plane)
            self.den_likelihood.append(likelihood)

        self.app_rec_plane = []
        self.app_likelihood = []
        for idx_plane in range(len(self.app_plane)):
            rec_plane, likelihood = \
                self.feature_compression_via_feat_coder(self.map_fn(self.app_plane[idx_plane]),
                                                        app_feat_codec,
                                                        mode=mode)
            self.app_rec_plane.append(rec_plane)
            self.app_likelihood.append(likelihood)

        ret_dict = {
            "den":{
                "rec_planes": self.den_rec_plane,
                "rec_likelihood": self.den_likelihood
            },
            "app":{
                "rec_planes": self.app_rec_plane,
                "rec_likelihood": self.app_likelihood
            }
        }

        return ret_dict # rec_planes & rates from density and appearance

    def decode_single_plane(self, y, z, feat_codec, target_plane_size, mode="train"): # mode: "train" or "eval"
        h, w = target_plane_size[2], target_plane_size[3]
        # pad to allow 6 strides of 2
        pad, unpad = compute_padding(h, w, min_div=2 ** 6)

        if mode == "train":
            out_net = feat_codec.forward(y, z)
        elif mode == "eval":
            out_enc = feat_codec.compress(y, z)
            out_dec = feat_codec.decompress(out_enc["strings"], out_enc["shape"])
            # pdb.set_trace()
            out_net = out_dec
            out_net.update({"strings_length": sum(len(s[0]) for s in out_enc["strings"])})

        # get rid of padding
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_net["x_hat"] = out_net["x_hat"].reshape([1, -1, h, w])

        return out_net

    def decode_all_planes(self, mode="train"):
        self.map_fn = torch.nn.Tanh()
        self.den_rec_plane = []
        self.den_likelihood = []
        for idx_plane in range(len(self.density_plane)):
            out_net = \
                self.decode_single_plane(self.den_latent_y[idx_plane],
                                         self.den_latent_z[idx_plane],
                                         self.den_feat_codec,
                                         self.density_plane[idx_plane].size(),
                                         mode=mode)
            # pdb.set_trace()
            self.den_rec_plane.append(out_net["x_hat"])
            self.den_likelihood.append(out_net)

        self.app_rec_plane = []
        self.app_likelihood = []
        for idx_plane in range(len(self.app_plane)):
            out_net = \
                self.decode_single_plane(self.app_latent_y[idx_plane],
                                         self.app_latent_z[idx_plane],
                                         self.app_feat_codec,
                                         self.app_plane[idx_plane].size(),
                                         mode=mode)
            # pdb.set_trace()
            self.app_rec_plane.append(out_net["x_hat"])
            self.app_likelihood.append(out_net)

        ret_dict = {
            "den": {
                "rec_planes": self.den_rec_plane,
                "rec_likelihood": self.den_likelihood
            },
            "app": {
                "rec_planes": self.app_rec_plane,
                "rec_likelihood": self.app_likelihood
            }
        }
        return ret_dict

    def enable_vec_qat(self):
        if self.vec_qat:
            print("Vector, QAT, bitwidth: 8 bits")
            self.q_fn = lambda x: qfn2.apply(x, 8)
        else:
            self.q_fn = torch.nn.Identity()
        # pdb.set_trace()

    def compute_densityfeature(self, xyz_sampled):
        # tanh = torch.nn.Tanh()
        # self.map_fn = torch.nn.Tanh() # default after submission
        self.map_fn = torch.nn.Identity() # default before submission

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        if self.additional_vec:
            add_coordinate_line = torch.stack((xyz_sampled[..., 0], xyz_sampled[..., 1], xyz_sampled[..., 2]))
            add_coordinate_line = torch.stack((torch.zeros_like(add_coordinate_line), add_coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        self.density_likelihood_list = []
        self.rec_density_feat_list = []
        for idx_plane in range(len(self.density_plane)):
            # pdb.set_trace()
            if self.compression:
                # rec_plane, likelihood = self.feature_compression(self.density_plane[idx_plane], self.return_likelihoods)
                if self.compression_strategy == 'batchwise_img_coding':
                    rec_plane, likelihood = self.batchwise_feature_compression(self.density_plane[idx_plane])
                elif self.compression_strategy == 'adaptor_feat_coding':
                    if self.compress_before_volrend:
                        rec_plane = self.den_rec_plane[idx_plane]
                    # else:
                    #     rec_plane, likelihood = self.feature_compression_via_feat_coder(self.map_fn(self.density_plane[idx_plane]),
                    #                                                                     self.den_feat_codec,
                    #                                                                     mode=self.mode)
                    #     if likelihood is not None:
                    #         self.density_likelihood_list.append(likelihood)
                    #     self.rec_density_feat_list.append(rec_plane)

                if self.using_external_codec:
                    rec_plane = self.den_rec_plane[idx_plane]

                plane_coef_point = F.grid_sample(rec_plane,
                                                 coordinate_plane[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_sampled.shape[:1])

                if self.additional_vec:
                    mat_id_0, mat_id_1 = self.matMode[idx_plane]
                    fst_add_line_coef_point = F.grid_sample(self.q_fn(self.map_fn(self.additional_density_line[mat_id_0])),
                                                     add_coordinate_line[[mat_id_0]],
                                                     align_corners=True).view(-1, *xyz_sampled.shape[:1])
                    sec_add_line_coef_point = F.grid_sample(self.q_fn(self.map_fn(self.additional_density_line[mat_id_1])),
                                                         add_coordinate_line[[mat_id_1]],
                                                         align_corners=True).view(-1, *xyz_sampled.shape[:1])

                    add_feat = fst_add_line_coef_point * sec_add_line_coef_point
                    plane_coef_point += add_feat


            else:
                plane_coef_point = F.grid_sample(self.map_fn(self.density_plane[idx_plane]), coordinate_plane[[idx_plane]],
                                                    align_corners=True).view(-1, *xyz_sampled.shape[:1])

            if self.compression:
                line_coef_point = F.grid_sample(self.q_fn(self.map_fn(self.density_line[idx_plane])),
                                                coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            else:
                line_coef_point = F.grid_sample(self.map_fn(self.density_line[idx_plane]),
                                                coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_appfeature(self, xyz_sampled):
        # tanh = torch.nn.Tanh()
        self.map_fn = torch.nn.Tanh()

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        if self.additional_vec:
            add_coordinate_line = torch.stack((xyz_sampled[..., 0], xyz_sampled[..., 1], xyz_sampled[..., 2]))
            add_coordinate_line = torch.stack((torch.zeros_like(add_coordinate_line), add_coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        self.app_likelihood_list = []
        self.rec_app_feat_list = []
        for idx_plane in range(len(self.app_plane)):
            if self.compression:
                # rec_plane, likelihood = self.feature_compression(self.app_plane[idx_plane], self.return_likelihoods)
                if self.compression_strategy == 'batchwise_img_coding':
                    rec_plane, likelihood = self.batchwise_feature_compression(self.app_plane[idx_plane])
                elif self.compression_strategy == 'adaptor_feat_coding':
                    if self.compress_before_volrend:
                        rec_plane = self.app_rec_plane[idx_plane]
                    # else:
                    #     rec_plane, likelihood = self.feature_compression_via_feat_coder(self.map_fn(self.app_plane[idx_plane]), self.app_feat_codec, mode=self.mode)
                    #     self.rec_app_feat_list.append(rec_plane)

                    #     if likelihood is not None:
                    #         self.app_likelihood_list.append(likelihood)

                if self.using_external_codec:
                    rec_plane = self.app_rec_plane[idx_plane]

                plane_coef_point.append(F.grid_sample(rec_plane, coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))

                if self.additional_vec:
                    # fst_axis, sec_axis = self.matMode[idx_plane]
                    mat_id_0, mat_id_1 = self.matMode[idx_plane]
                    fst_add_line_coef_point = F.grid_sample(self.q_fn(self.map_fn(self.additional_app_line[mat_id_0])),
                                                            add_coordinate_line[[mat_id_0]],
                                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
                    sec_add_line_coef_point = F.grid_sample(self.q_fn(self.map_fn(self.additional_app_line[mat_id_1])),
                                                            add_coordinate_line[[mat_id_1]],
                                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])

                    add_feat = fst_add_line_coef_point * sec_add_line_coef_point
                    plane_coef_point[-1] += add_feat

                # plane_coef_point.append(F.grid_sample(self.feature_compression(self.app_plane[idx_plane]), coordinate_plane[[idx_plane]],
                #                                 align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            else:
                plane_coef_point.append(F.grid_sample(self.map_fn(self.app_plane[idx_plane]), coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            if self.compression:
                line_coef_point.append(F.grid_sample(self.q_fn(self.map_fn(self.app_line[idx_plane])),
                                                     coordinate_line[[idx_plane]],
                                                     align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            else:
                line_coef_point.append(F.grid_sample(self.map_fn(self.app_line[idx_plane]),
                                                     coordinate_line[[idx_plane]],
                                                     align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    def forward_densityfeature(self, xyz_sampled, plane_feature=None, line_feature=None):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        # pdb.set_trace()
        for idx_plane in range(3):
            plane_coef_point = F.grid_sample(plane_feature[idx_plane], coordinate_plane[[idx_plane]],
                                                    align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(line_feature[idx_plane], coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def forward_appfeature(self, xyz_sampled, plane_feature=None, line_feature=None):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]

        for idx_plane in range(3):
            plane_coef_point.append(F.grid_sample(plane_feature[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(line_feature[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        # pdb.set_trace()
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(vector_comps[idx].view(n_comp, n_size),
                                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[
                                                                                                      idx]))  # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2  # + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2  # + reg(self.app_line[idx]) * 1e-3
        return total

    def forward_with_feature(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1,
                                   plane_feature=None, line_feature=None, alphaMask=None):
        # load scene specific property
        # pdb.set_trace()
        self.alphaMask = alphaMask

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        # mask out invaild samples
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)

            sigma_feature = self.forward_densityfeature(xyz_sampled[ray_valid],
                                                        plane_feature["den"],
                                                        line_feature["den"])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.forward_appfeature(xyz_sampled[app_mask],
                                                   plane_feature["app"],
                                                   line_feature["app"])

            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map  # rgb, sigma, alpha, weight, bg_weight


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.app_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
    

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)


        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total