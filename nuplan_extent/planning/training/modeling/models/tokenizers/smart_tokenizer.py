import torch
import torch.nn as nn
import pickle
import numpy as np


class SMARTTokenizer(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                 map_token_traj_path: str) -> None:
        super().__init__()
        self.noise = True

        self.map_token_traj_path = map_token_traj_path
        self.init_map_token()

    def tokenize_data(self, data):
        data['heterodata'].data = self.match_token_map(data['heterodata'].data)
        data['heterodata'].data = self.sample_pt_pred(data['heterodata'].data)
        return data

    def init_map_token(self):
        self.argmin_sample_len = 3
        map_token_traj = pickle.load(open(self.map_token_traj_path, 'rb'))
        self.map_token = {'traj_src': map_token_traj['traj_src'], }
        traj_end_theta = np.arctan2(self.map_token['traj_src'][:, -1, 1]-self.map_token['traj_src'][:, -2, 1],
                                    self.map_token['traj_src'][:, -1, 0]-self.map_token['traj_src'][:, -2, 0])
        indices = torch.linspace(0, self.map_token['traj_src'].shape[1]-1, steps=self.argmin_sample_len).cpu().long()
        self.map_token['sample_pt'] = torch.from_numpy(self.map_token['traj_src'][:, indices]).to(torch.float)
        self.map_token['traj_end_theta'] = torch.from_numpy(traj_end_theta).to(torch.float)
        self.map_token['traj_src'] = torch.from_numpy(self.map_token['traj_src']).to(torch.float)

    def forward(self, input):
        pass

    def match_token_map(self, data):
        traj_pos = data['map_save']['traj_pos'].to(torch.float) # N, 3, 2 (x,y last)
        traj_theta = data['map_save']['traj_theta'].to(torch.float)
        pl_idx_list = data['map_save']['pl_idx_list']
        token_sample_pt = self.map_token['sample_pt'].to(traj_pos.device)
        token_src = self.map_token['traj_src'].to(traj_pos.device)
        max_traj_len = self.map_token['traj_src'].shape[1]
        pl_num = traj_pos.shape[0]

        pt_token_pos = traj_pos[:, 0, :].clone()
        pt_token_orientation = traj_theta.clone()
        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = -sin
        rot_mat[..., 1, 0] = sin
        rot_mat[..., 1, 1] = cos
        traj_pos_local = torch.bmm((traj_pos - traj_pos[:, 0:1]), rot_mat.view(-1, 2, 2))
        distance = torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1))**2, dim=(-2, -1))
        pt_token_id = torch.argmin(distance, dim=1)

        if self.noise:
            topk_indices = torch.argsort(torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1))**2, dim=(-2, -1)), dim=1)[:, :8]
            sample_topk = torch.randint(0, topk_indices.shape[-1], size=(topk_indices.shape[0], 1), device=topk_indices.device)
            pt_token_id = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)

        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = sin
        rot_mat[..., 1, 0] = -sin
        rot_mat[..., 1, 1] = cos
        token_src_world = torch.bmm(token_src[None, ...].repeat(pl_num, 1, 1, 1).reshape(pl_num, -1, 2),
                                    rot_mat.view(-1, 2, 2)).reshape(pl_num, token_src.shape[0], max_traj_len, 2) + traj_pos[:, None, [0], :]
        token_src_world_select = token_src_world.view(-1, 1024, 11, 2)[torch.arange(pt_token_id.view(-1).shape[0]), pt_token_id.view(-1)].view(pl_num, max_traj_len, 2)

        pl_idx_full = pl_idx_list.clone()
        token2pl = torch.stack([torch.arange(len(pl_idx_list), device=traj_pos.device), pl_idx_full.long()])
        count_nums = []
        for pl in pl_idx_full.unique():
            pt = token2pl[0, token2pl[1, :] == pl]
            left_side = (data['pt_token']['side'][pt] == 0).sum()
            right_side = (data['pt_token']['side'][pt] == 1).sum()
            center_side = (data['pt_token']['side'][pt] == 2).sum()
            count_nums.append(torch.Tensor([left_side, right_side, center_side]))
        count_nums = torch.stack(count_nums, dim=0)
        num_polyline = int(count_nums.max().item())
        traj_mask = torch.zeros((int(len(pl_idx_full.unique())), 3, num_polyline), dtype=bool)
        idx_matrix = torch.arange(traj_mask.size(2)).unsqueeze(0).unsqueeze(0)
        idx_matrix = idx_matrix.expand(traj_mask.size(0), traj_mask.size(1), -1)  #
        counts_num_expanded = count_nums.unsqueeze(-1)
        mask_update = idx_matrix < counts_num_expanded
        traj_mask[mask_update] = True

        data['pt_token']['traj_mask'] = traj_mask
        data['pt_token']['position'] = torch.cat([pt_token_pos, torch.zeros((data['pt_token']['num_nodes'], 1),
                                                                            device=traj_pos.device, dtype=torch.float)], dim=-1)
        data['pt_token']['orientation'] = pt_token_orientation
        data['pt_token']['height'] = data['pt_token']['position'][:, -1]
        data[('pt_token', 'to', 'map_polygon')] = {}
        data[('pt_token', 'to', 'map_polygon')]['edge_index'] = token2pl
        data['pt_token']['token_idx'] = pt_token_id
        return data

    def sample_pt_pred(self, data):
        traj_mask = data['pt_token']['traj_mask']
        raw_pt_index = torch.arange(1, traj_mask.shape[2]).repeat(traj_mask.shape[0], traj_mask.shape[1], 1)
        masked_pt_index = raw_pt_index.view(-1)[torch.randperm(raw_pt_index.numel())[:traj_mask.shape[0]*traj_mask.shape[1]*((traj_mask.shape[2]-1)//3)].reshape(traj_mask.shape[0], traj_mask.shape[1], (traj_mask.shape[2]-1)//3)]
        masked_pt_index = torch.sort(masked_pt_index, -1)[0]
        pt_valid_mask = traj_mask.clone()
        pt_valid_mask.scatter_(2, masked_pt_index, False)
        pt_pred_mask = traj_mask.clone()
        pt_pred_mask.scatter_(2, masked_pt_index, False)
        tmp_mask = pt_pred_mask.clone()
        tmp_mask[:, :, :] = True
        tmp_mask.scatter_(2, masked_pt_index-1, False)
        pt_pred_mask.masked_fill_(tmp_mask, False)
        pt_pred_mask = pt_pred_mask * torch.roll(traj_mask, shifts=-1, dims=2)
        pt_target_mask = torch.roll(pt_pred_mask, shifts=1, dims=2)

        data['pt_token']['pt_valid_mask'] = pt_valid_mask[traj_mask]
        data['pt_token']['pt_pred_mask'] = pt_pred_mask[traj_mask]
        data['pt_token']['pt_target_mask'] = pt_target_mask[traj_mask]

        return data