import torch
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.loss import soft_dtw
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.loss import path_soft_dtw

def dilate_loss(outputs, targets, device, alpha=1,gamma=0.01):
	# 修改点 1: 适配输入形状 (B, 1, L)
	# 将形状从 (Batch, 1, Length) 转置为 (Batch, Length, 1)
	# 如果已经是 (B, L, 1) 则不受影响，这里假设用户严格传入 (B, 1, L)
	if outputs.shape[1] == 1 and outputs.shape[2] > 1:
		outputs = outputs.transpose(1, 2)  # 变为 [B, L, 1]
		targets = targets.transpose(1, 2)  # 变为 [B, L, 1]
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk
	loss_shape = softdtw_batch(D,gamma)

	loss_temporal = 0
	if alpha != 1:
		path_dtw = path_soft_dtw.PathDTWBatch.apply
		path = path_dtw(D,gamma)
		idx = torch.arange(1, N_output + 1, device=device).float().view(N_output, 1)
		Omega = soft_dtw.pairwise_distances(idx)
		loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output)
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss, loss_shape, loss_temporal
