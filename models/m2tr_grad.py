import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

class M2TRGradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        self.features = None
        self.grads = None

        self.forward_hook = target_layer.register_forward_hook(self._save_features)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.grads = grad_output[0].detach()

    def generate_cam(self, x, target_class=None):
        img_tensor = x["img"]
        bsz, _, img_h, img_w = img_tensor.shape
        cam_list = []
        for i in range(bsz):
            output = self.model(img_tensor[i:i+1])
            logits = output["logits"]
            print(logits)
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()

            # 提取标量得分：第0个样本、第target_class个通道 → 0维标量
            target_score = logits[0, target_class]

            # 反向传播（标量可正常backward）
            self.model.zero_grad()
            target_score.backward(retain_graph=True)

            feat_map = self.features
            grad_map = self.grads

            grad_1 = grad_map
            grad_2 = grad_1 ** 2
            grad_3 = grad_2 * grad_1

            sum_feat = torch.sum(feat_map, dim=(2, 3), keepdim=True)
            denominator = 2 * grad_2 + sum_feat * grad_3 + 1e-8

            alpha = grad_2 / denominator
            alpha = torch.mean(alpha, dim=(2, 3), keepdim=True)

            cam = torch.sum(alpha * feat_map, dim=1, keepdim=True)
            cam = torch.relu(cam)

            cam = cam.squeeze()
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-8)
            cam = cam.cpu().numpy()

            cam = cv2.resize(cam, (img_w, img_h))
            cam_list.append(cam)
        return cam_list

    def remove_hooks(self):
        """移除钩子，避免内存泄漏"""
        self.forward_hook.remove()
        self.backward_hook.remove()

def overlay_heatmap(batch_id, samples, cam_list, alpha=0.5, colormap=cv2.COLORMAP_JET):
    for id, (image_rgb, cam) in enumerate(zip(samples["img"], cam_list)):
        h, w = image_rgb.shape[:2]
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255.0

        img_norm = np.transpose(np.float32(image_rgb.cpu().numpy()), (1, 2, 0))
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
        vis = heatmap * alpha + img_norm
        vis = vis / np.max(vis)
        # return np.uint8(255 * vis)
        cv2.imwrite(f"grad_cam/{batch_id}_{id}.jpg", cv2.cvtColor(np.uint8(255 * vis), cv2.COLOR_RGB2BGR))

