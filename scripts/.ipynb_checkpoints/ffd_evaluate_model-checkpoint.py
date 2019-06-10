import argparse
import os
import torch

from attrdict import AttrDict
import matplotlib.pyplot as plt

from sgan.data.loader import data_loader
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='checkpoint_with_model.pt', type=str)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--GAN_type', default='ff')
parser.add_argument('--dest', default='random.txt', type=str)

torch.manual_seed(0)
def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        rep_dim=args.rep_dim,
        mlp_dim=args.mlp_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        pos_embed=args.pos_embed,
        pos_embed_freq=args.pos_embed_freq,
        )

    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

# ,
#         pos_embed=args.pos_embed,
#         pos_embed_freq=args.pos_embed_freq

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    i = 0
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.75,0.75])
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            i = i + 1
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            
            # print("Trajs: ")
            # print(obs_traj[:,0,:])
            # print(obs_traj_rel[:,0,:])
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            print(num_samples)
            for rr in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                pred_traj_fake_plot = pred_traj_fake.permute(1, 0, 2)
                pred_traj_fake_plot_single = pred_traj_fake_plot[0,:,:].cpu().numpy() 
                # print(pred_traj_fake_plot_single.shape)
                if i < 25:
                    pred_traj_fake_permuted = pred_traj_fake.permute(1, 0, 2)
                    pred_traj_gt_permuted = pred_traj_gt.permute(1, 0, 2)
                    obs_traj_permuted = obs_traj.permute(1, 0, 2)

                    # if k == 0:
                    #     view_traj(ax, pred_traj_fake_permuted[0,:,:], pred_traj_gt_permuted[0,:,:], obs_traj_permuted[0,:,:], args, all_three=True)
                    # else:
                    yy = 0
                    view_traj(ax, pred_traj_fake_permuted[yy,:,:], pred_traj_gt_permuted[yy,:,:], obs_traj_permuted[yy,:,:], args)

                # plt.plot(pred_traj_fake_plot_single[:,0], pred_traj_fake_plot_single[:,1])
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            # plt.xlim((0,17))
            # plt.ylim((-10,10))
            # plt.legend()
            if i < 25:
                plt.show()
                plt.cla()

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def evaluate_attn(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    i = 0
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.75,0.75])
    generator.eval()
    for batch in loader:
        i = i + 1
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         non_linear_ped, loss_mask, seq_start_end) = batch
        
        # print("Trajs: ")
        # print(obs_traj[:,0,:])
        # print(obs_traj_rel[:,0,:])
        ade, fde = [], []
        total_traj += pred_traj_gt.size(1)

        for rr in range(1):
            if i == 1:
                obs_traj_rel.requires_grad_(True)
                obs_traj.requires_grad_(True)
            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )

            if i == 1:
                print("Evaluating Attention")
                # get_attn(pred_traj_fake_rel, obs_traj_rel)
                print("Getting Attention")
                pred_traj_zero = pred_traj_fake_rel[:,0,:]
                obs_traj_zero = obs_traj_rel[:,0,:]
                print(pred_traj_zero.shape)
                print(obs_traj_zero.shape)
                print(pred_traj_zero[6,0])
                print(pred_traj_zero.requires_grad)
                print(obs_traj_zero.requires_grad)
                obs_traj_zero_grad = torch.autograd.grad(pred_traj_zero[6,0], obs_traj_rel)
                print(obs_traj_zero_grad[0][:,0,:].abs().sum(1))



            pred_traj_fake_plot = pred_traj_fake.permute(1, 0, 2)
            pred_traj_fake_plot_single = pred_traj_fake_plot[0,:,:].detach().cpu().numpy() 
            # print(pred_traj_fake_plot_single.shape)
            if i < 2:
                pred_traj_fake_permuted = pred_traj_fake.permute(1, 0, 2)
                pred_traj_gt_permuted = pred_traj_gt.permute(1, 0, 2)
                obs_traj_permuted = obs_traj.permute(1, 0, 2)

                # if k == 0:
                #     view_traj(ax, pred_traj_fake_permuted[0,:,:], pred_traj_gt_permuted[0,:,:], obs_traj_permuted[0,:,:], args, all_three=True)
                # else:
                yy = 0
                view_traj(ax, pred_traj_fake_permuted[yy,:,:], pred_traj_gt_permuted[yy,:,:], obs_traj_permuted[yy,:,:], args)
                plt.legend()
                plt.show()
                plt.cla()

            # plt.plot(pred_traj_fake_plot_single[:,0], pred_traj_fake_plot_single[:,1])
            ade.append(displacement_error(
                pred_traj_fake, pred_traj_gt, mode='raw'
            ))
            fde.append(final_displacement_error(
                pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
            ))

        ade_sum = evaluate_helper(ade, seq_start_end)
        fde_sum = evaluate_helper(fde, seq_start_end)

        ade_outer.append(ade_sum)
        fde_outer.append(fde_sum)
        # plt.xlim((0,17))
        # plt.ylim((-10,10))
        # plt.legend()
        # if i < 2:


    ade = sum(ade_outer) / (total_traj * args.pred_len)
    fde = sum(fde_outer) / (total_traj)
    return ade, fde

def get_attn(pred_traj_fake_rel, obs_traj_rel):
    print("Getting Attention")
    pred_traj_zero = pred_traj_fake_rel.permute(1,0,2)[0,:,:]
    obs_traj_zero = obs_traj_rel.permute(1,0,2)[0,:,:]
    print(pred_traj_zero.shape)
    print(obs_traj_zero.shape)
    print(pred_traj_zero[0,0])
    print(pred_traj_zero.requires_grad)
    print(obs_traj_zero.requires_grad)
    obs_traj_zero_grad = torch.autograd.grad(pred_traj_zero[0,0], obs_traj_zero, allow_unused=True)
    print(obs_traj_zero_grad)
    return

def view_traj(ax, fake_pred, real_pred, obs, args, all_three=False):
    fake_pred = fake_pred.detach().cpu().numpy()
    real_pred = real_pred.detach().cpu().numpy()
    obs = obs.detach().cpu().numpy()

    # fake_traj = np.concatenate((obs, fake_pred), axis=0)
    # real_traj = np.concatenate((obs, real_pred), axis=0)
    
    
 
    ax.plot(fake_pred[:,0], fake_pred[:,1], c='g', label='Predicted')
    ax.plot(obs[:,0],  obs[:,1],  'b',  label='Observed')
    ax.plot(real_pred[:,0], real_pred[:,1], 'r', label='Real Pred')
    # if args.dataset_name != 'straight':
    #     ax.plot(x_pred[1,:], y_pred[1,:], 'r', label='Real Pred 2')
    #     ax.plot(x_pred[2,:], y_pred[2,:], 'r', label='Real Pred 3')
    #     if args.dataset_name != 'threeTraj' and args.dataset_name != 'uneqthreeTraj':
    #         ax.plot(x_pred[3,:], y_pred[3,:], 'r', label='Real Pred 4')
    #         ax.plot(x_pred[4,:], y_pred[4,:], 'r', label='Real Pred 5')
    #         if args.dataset_name != 'fiveTraj':
    #             ax.plot(x_pred[5,:], y_pred[5,:], 'r', label='Real Pred 4')
    #             ax.plot(x_pred[6,:], y_pred[6,:], 'r', label='Real Pred 5')


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    global TrajectoryGenerator, TrajectoryDiscriminator
    if args.GAN_type == 'rnn':
        print("Default Social GAN")
        from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
    elif args.GAN_type == 'simple_rnn':
        print("Default Social GAN")
        from sgan.rnn_models import TrajectoryGenerator, TrajectoryDiscriminator
    else:
        print("Feedforward GAN")
        from sgan.ffd_models import TrajectoryGenerator, TrajectoryDiscriminator


    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        print(_args)
        path = get_dset_path(_args.dataset_name, args.dset_type)
        print(path)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        # result_str = '\n GAN_type: {}, Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f} \n'.format(
        #              _args.GAN_type, _args.dataset_name, _args.pred_len, ade, fde)
        result_str = 'GAN_type: {}, Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f} Samples: {} \n \n'.format(
                      args.GAN_type, _args.dataset_name, _args.pred_len, ade, fde, args.num_samples)

        print(result_str)
        with open(args.dest, "a") as myfile:
            myfile.write(result_str)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
