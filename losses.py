import kornia
import torch as T
import torch.nn.functional as F
from torch_cluster import knn_graph

import utils_brush
from vgg import VGG19


class StyleTransferLosses(VGG19):
    def __init__(self, weight_file, content_img: T.Tensor, style_img: T.Tensor, 
                 content_layers, style_layers,
                 scale_by_y=False, content_weights=None, style_weights=None):
        super(StyleTransferLosses, self).__init__(weight_file)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.scale_by_y = scale_by_y

        content_weights = content_weights if content_weights is not None else [
            1.] * len(self.content_layers)
        style_weights = style_weights if style_weights is not None else [
            1.] * len(self.style_layers)
        self.content_weights = {}
        self.style_weights = {}

        content_features = content_img
        style_features = style_img
        self.content_features = {}
        self.style_features = {}
        if scale_by_y:
            self.weights = {}

        i, j = 0, 0
        self.to(content_img.device)
        with T.no_grad():
            for name, layer in self.named_children():
                content_features = layer(content_features)
                style_features = layer(style_features)
                if name in content_layers:
                    self.content_features[name] = content_features
                    if scale_by_y:
                        self.weights[name] = T.minimum(
                            content_features, T.sigmoid(content_features))

                    self.content_weights[name] = content_weights[i]
                    i += 1

                if name in style_layers:
                    self.style_features[name] = utils_brush.gram_matrix(
                        style_features)
                    self.style_weights[name] = style_weights[j]
                    j += 1

    def forward(self, input, mask=None):
        content_loss, style_loss = 0., 0.
        features = input
        for name, layer in self.named_children():
            features = layer(features)
            if mask is not None:
                b, c, h, w = features.shape
                now_mask = kornia.geometry.transform.resize(
                    mask, (h, w)
                )

            if name in self.content_layers:
                loss = (features - self.content_features[name])
                if self.scale_by_y:
                    loss *= self.weights[name]

                if mask is not None:
                    loss *= now_mask[:, 0:1, :, :]
                    if T.sum(now_mask[:, 0:1, :, :]) > 0:
                        content_loss += (T.sum(loss ** 2) / T.sum(now_mask[:, 0:1, :, :]) *
                                         self.content_weights[name])

                else:
                    content_loss += (T.mean(loss ** 2) *
                                     self.content_weights[name])

            if name in self.style_layers:
                if mask is None:
                    now_gram = utils_brush.gram_matrix(features)
                else:
                    now_mask = now_mask.view(1 * 3, h * w)
                    flatten_features = features.view(b * c, h * w)
                    flatten_features = flatten_features[:, now_mask[0, :] > 0]
                    ch, length = flatten_features.shape
                    if length == 0:
                        now_gram = None
                    else:
                        now_gram = T.mm(flatten_features, flatten_features.t())
                        now_gram = now_gram.div(ch * length)

                if now_gram is not None:
                    loss = F.mse_loss(
                        self.style_features[name],
                        now_gram,
                        reduction='sum')
                    if not T.isinf(loss):
                        style_loss += (loss * self.style_weights[name])
                    else:
                        print(loss, name)

        return content_loss, style_loss


def total_variation_loss(
        location: T.Tensor, curve_s: T.Tensor, curve_e: T.Tensor, K=10):
    se_vec = curve_e - curve_s
    x_nn_idcs = knn_graph(location, k=K)[0]
    x_sig_nns = se_vec[x_nn_idcs].view(
        *((se_vec.shape[0], K) + se_vec.shape[1:]))
    dist_to_centroid = T.mean(T.sum((utils_brush.projection(
        x_sig_nns) - utils_brush.projection(se_vec)[..., None, :]) ** 2, dim=-1))
    return dist_to_centroid


def curvature_loss(curve_s: T.Tensor, curve_e: T.Tensor, curve_c: T.Tensor):
    v1 = curve_s - curve_c
    v2 = curve_e - curve_c
    dist_se = T.norm(curve_e - curve_s, dim=-1) + 1e-6
    return T.mean(T.norm(v1 + v2, dim=-1) / dist_se)


def tv_loss(x):
    diff_i = T.mean((x[..., :, 1:] - x[..., :, :-1]) ** 2)
    diff_j = T.mean((x[..., 1:, :] - x[..., :-1, :]) ** 2)
    loss = diff_i + diff_j
    return loss
