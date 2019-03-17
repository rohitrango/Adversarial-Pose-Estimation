'''
file containing loss functions
'''
import torch

def get_loss_recon(out, inp, mode):
	'''
	Get the reconstruction loss
	'''
	if mode == 'mse':
		loss = ((out - inp)**2).mean()
	else:
		raise NotImplementedError


def get_loss_disc(output, discriminator, detach=False, real=True):
	'''
	Get discriminator loss
	'''
	outs = output.detach() if detach else output
	if real:
		loss = torch.log(discriminator(outs)).mean()
	else:
		loss = torch.log(1 - discriminator(outs)).mean()
	return loss


def gen_loss(ground_truth,
			 outputs,
			 pose_discriminator,
			 conf_discriminator,
			 mode='mse',
			 alpha=1/220.,
			 beta=1/180.
			):
	'''
	Get generator loss
	'''
	gt_maps = torch.cat([ground_truth['heatmaps'], ground_truth['occlusions']], 1)
	loss_recon = 0.0
	loss_conf_disc = 0.0
	loss_pose_disc = 0.0
	for output in outputs:
		loss_recon = loss_recon + get_loss_recon(output, gt_maps, mode)
		loss_conf_disc = loss_conf_disc + get_loss_disc(output, conf_discriminator)
		loss_pose_disc = loss_pose_disc + get_loss_disc(output, pose_discriminator)

	loss_recon = loss_recon / len(outputs)
	loss_conf_disc = loss_conf_disc / len(outputs)
	loss_pose_disc = loss_pose_disc / len(outputs)
	# Add discriminator loss
	return {
		'loss': loss_recon + alpha*loss_conf_disc + beta*loss_pose_disc,
		'recon': loss_recon,
		'conf_disc': loss_conf_disc,
		'pose_disc': loss_pose_disc,
	}


def dis_loss(ground_truth,
			 outputs,
			 pose_discriminator,
			 conf_discriminator,
			 alpha=1/220.0,
			 beta=1/180.0
			):
	'''
	Get discriminator loss
	'''
	gt_maps = torch.cat([ground_truth['heatmaps'], ground_truth['occlusions']], 1)
	loss_conf_real = get_loss_disc(gt_maps, conf_discriminator)
	loss_pose_real = get_loss_disc(gt_maps, pose_discriminator)
	# False for generator
	for output in outputs:
		loss_conf_disc = loss_conf_disc + get_loss_disc(output, conf_discriminator, detach=False, real=False)
		loss_pose_disc = loss_pose_disc + get_loss_disc(output, pose_discriminator, detach=False, real=False)

	loss_conf_disc = loss_conf_disc / len(outputs)
	loss_pose_disc = loss_pose_disc / len(outputs)
	# Add discriminator loss
	return {
		'loss': (loss_conf_real + loss_conf_disc + loss_pose_real + loss_pose_disc),
		'conf_disc_real': loss_conf_real,
		'pose_disc_real': loss_pose_real,
		'conf_disc_fake': loss_conf_disc,
		'pose_disc_fake': loss_pose_disc,
	}
