import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict
import matplotlib.pyplot as plt 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from helper_FF import generateTraj

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

sys.path.append(os.path.join('teacher'))

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--num_epochs', default=0, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=8, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=32, type=int)


# Generator Options
parser.add_argument('--encoder_h_dim_g', default=16, type=int)
parser.add_argument('--decoder_h_dim_g', default=16, type=int)
parser.add_argument('--noise_dim', default=(4,), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global')
parser.add_argument('--clipping_threshold_g', default=2.0, type=float)
parser.add_argument('--g_learning_rate', default=0.0001, type=float)
parser.add_argument('--g_steps', default=1, type=int)
#g = 0.0001

# Pooling Options
parser.add_argument('--pooling_type', default=None)
parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=8, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=32, type=int)
parser.add_argument('--d_learning_rate', default=0.001, type=float)
parser.add_argument('--d_steps', default=1, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)
#d = 0.001

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--checkpoint_every', default=3000000, type=int)
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

# Important changing Options
parser.add_argument('--GAN_type', default='ff')
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--restore_from_checkpoint', default=0, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
# Loss Options
parser.add_argument('--l2_loss_weight', default=0.5, type=float)
parser.add_argument('--best_k', default=20, type=int)
parser.add_argument('--controlled_expt', default=0, type=bool_flag)
parser.add_argument('--pos_embed', default=0, type=bool_flag)
parser.add_argument('--pos_embed_freq', default=100, type=int)


#Teacher 
parser.add_argument("--mode", default="training", help="mode options: training | refinement | testing")
# parser.add_argument('--restore_from_checkpoint', default=1, type=int)
# parser.add_argument('--checkpoint_every', default=300000, type=int)
parser.add_argument("--teacher-name", default="gpurollout", help="teacher options: default | gpurollout")
parser.add_argument("--rollout_steps", type=int, default=50000, help= "Roll Out Steps. [100]")
parser.add_argument("--rollout_rate", type=float, default=10, help="Roll Out Rate [50]")
parser.add_argument("--rollout_method", default="momentum", help="Rollout Method: sgd | momentum")
parser.add_argument("--use_refined", default=True, help="True for shaping using refined samples, False for using default samples [False]")
parser.add_argument("--epoch", default=5, help="Epoch to train G and D [20]/ Epochs to refine D [5]")
parser.add_argument("--load_epoch", default=0, help="Epoch to load from for refinement")
parser.add_argument("--load_model_dir", default="dc_checkpoints/celebA/epoch_5_teacher_default_rollout_method_momentum_rollout_steps_100_rollout_rate_50.00000/celebA_64_64_64/", help="directory to load model from")
parser.add_argument("--refine_D_iters", default=1, help="Number of iteration to refine D [4]")
parser.add_argument("--save_figs", default=True, help="True for saving the comparison figures, False for nothing [False]")

parser.add_argument('--images_name', default='images_new', type=str)
parser.add_argument('--encoder_type', default='MLP', type=str)
parser.add_argument('--decoder_type', default='LSTM', type=str)

torch.manual_seed(777)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def main(args):
    if args.mode == 'training':
        args.checkpoint_every = 100
        args.teacher_name = "default"
        args.restore_from_checkpoint = 0
        #args.l2_loss_weight = 0.0
        args.rollout_steps = 1
        args.rollout_rate = 1
        args.rollout_method = 'sgd'
        #print("HHHH"+str(args.l2_loss_weight))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    global TrajectoryGenerator, TrajectoryDiscriminator
    if args.GAN_type == 'rnn':
        print("Default Social GAN")
        from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
    elif args.GAN_type == 'simple_rnn':
        print("Default Social GAN")
        from sgan.rnn_models import TrajectoryGenerator, TrajectoryDiscriminator
    else:
        print("Feedforward GAN")
        if(args.)
        from sgan.cgs_ffd_models import TrajectoryGenerator, TrajectoryDiscriminator

    #image_dir = 'images/' + 'curve_5_traj_l2_0.5'
    image_dir = 'images/' + args.images_name + '_' + str(args.l2_loss_weight)
    print(args.l2_loss_weight)
    print("Image Dir: ", image_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    # build teacher
    print("[!] teacher_name: ", args.teacher_name)

    if args.teacher_name == 'default':
        teacher = None 
    elif args.teacher_name == 'gpurollout':
        from teacher_gpu_rollout_torch import TeacherGPURollout
        teacher = TeacherGPURollout(args)
        teacher.set_env(discriminator, generator)
        print("GPU Rollout Teacher")
    else:
        raise NotImplementedError

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    # # Create D optimizer.
    # self.d_optim = tf.train.AdamOptimizer(self.disc_LR*config.D_LR, beta1=config.beta1)
    # # Compute the gradients for a list of variables.
    # self.grads_d_and_vars = self.d_optim.compute_gradients(self.d_loss, var_list=self.d_vars)
    # self.grad_default_real = self.d_optim.compute_gradients(self.d_loss_real, var_list=inputs)
    # # Ask the optimizer to apply the capped gradients.
    # self.update_d = self.d_optim.apply_gradients(self.grads_d_and_vars) 
    # ## Get Saliency Map - Teacher 
    # self.saliency_map = tf.gradients(self.d_loss, self.inputs)[0]

    # ###### G Optimizer ######
    # # Create G optimizer.
    # self.g_optim = tf.train.AdamOptimizer(config.learning_rate*config.G_LR, beta1=config.beta1)

    # # Compute the gradients for a list of variables.
    # ## With respect to Generator Weights - AutoLoss
    # self.grad_default = self.g_optim.compute_gradients(self.g_loss, var_list=[self.G, self.g_vars]) 
    # ## With Respect to Images given to D - Teacher
    # # self.grad_default = g_optim.compute_gradients(self.g_loss, var_list=)
    # if config.teacher_name == 'default':
    # self.optimal_grad = self.grad_default[0][0]
    # self.optimal_batch = self.G - self.optimal_grad 
    # else:
    # self.optimal_grad, self.optimal_batch = self.teacher.build_teacher(self.G, self.D_, self.grad_default[0][0], self.inputs)

    # # Ask the optimizer to apply the manipulated gradients.
    # grads_collected = tf.gradients(self.G, self.g_vars, self.optimal_grad)
    # grads_and_vars_collected = list(zip(grads_collected, self.g_vars))

    # self.g_teach = self.g_optim.apply_gradients(grads_and_vars_collected)


    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.75,0.75])

    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:

            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:

                if args.mode != 'testing':
                    step_type = 'd'
                    losses_d = discriminator_step(args, batch, generator,
                                                  discriminator, d_loss_fn,
                                                  optimizer_d, teacher, args.mode)
                    checkpoint['norm_d'].append(
                        get_total_norm(discriminator.parameters()))

                d_steps_left -= 1
            
            elif g_steps_left > 0:
                
                if args.mode != 'testing':
                    step_type = 'g'
                    losses_g = generator_step(args, batch, generator,
                                              discriminator, g_loss_fn,
                                              optimizer_g, args.mode)
                    checkpoint['norm_g'].append(
                        get_total_norm(generator.parameters())
                    )

                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every  == 0 and args.mode != 'testing':
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)
                
                # # Check stats on the validation set
                # logger.info('Checking stats on val ...')
                # metrics_val = check_accuracy(
                #     args, val_loader, generator, discriminator, d_loss_fn
                # )
                # logger.info('Checking stats on train ...')
                # metrics_train = check_accuracy(
                #     args, train_loader, generator, discriminator,
                #     d_loss_fn, limit=True
                # )

                # for k, v in sorted(metrics_val.items()):
                #     logger.info('  [val] {}: {:.3f}'.format(k, v))
                #     checkpoint['metrics_val'][k].append(v)
                # for k, v in sorted(metrics_train.items()):
                #     logger.info('  [train] {}: {:.3f}'.format(k, v))
                #     checkpoint['metrics_train'][k].append(v)

                # min_ade = min(checkpoint['metrics_val']['ade'])
                # min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                # if metrics_val['ade'] == min_ade:
                #     logger.info('New low for avg_disp_error')
                #     checkpoint['best_t'] = t
                #     checkpoint['g_best_state'] = generator.state_dict()
                #     checkpoint['d_best_state'] = discriminator.state_dict()

                # if metrics_val['ade_nl'] == min_ade_nl:
                #     logger.info('New low for avg_disp_error_nl')
                #     checkpoint['best_t_nl'] = t
                #     checkpoint['g_best_nl_state'] = generator.state_dict()
                #     checkpoint['d_best_nl_state'] = discriminator.state_dict()


            if t % 50 == 0:
                # save = False
                # if t == 160:
                    # save = True
                # print(t)
                plot_trajectory(fig, ax, args, val_loader, generator, teacher, args.mode, t, save=True, image_dir=image_dir)


            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                print("Iteration: ", t)
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)


                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break


def plot_trajectory(fig, ax, args, loader, generator, teacher, mode, t, image_dir=None, save=False):
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            
            for k in range(100):
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                # ##Guide traj fake
                if mode == 'refinement':
                    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
                    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
                    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
                    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
                    _, traj_fake_rel  = teacher.build_teacher(traj_fake_rel, traj_real_rel, seq_start_end)
                    pred_traj_fake_rel = traj_fake_rel[args.obs_len:]
                    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                pred_traj_fake_permuted = pred_traj_fake.permute(1, 0, 2)
                pred_traj_gt_permuted = pred_traj_gt.permute(1, 0, 2)
                obs_traj_permuted = obs_traj.permute(1, 0, 2)

                if k == 0:
                    view_traj(ax, pred_traj_fake_permuted[0,:,:], pred_traj_gt_permuted[0,:,:], obs_traj_permuted[0,:,:], args, all_three=True)
                else:
                    view_traj(ax, pred_traj_fake_permuted[0,:,:], pred_traj_gt_permuted[0,:,:], obs_traj_permuted[0,:,:], args)

            ax.legend()
            ax.set_ylim((-10, 10))
            ax.set_xlim((0, 20))
            plt.show()
            if save:
                print("Saving Plot")
                # plt.style.use('dark_background')
                path = image_dir + '/' + str(t) + '.jpg'
                fig.savefig(path, bbox_inches="tight")   
            ax.clear()
            break 
        return


def view_traj(ax, fake_pred, real_pred, obs, args, all_three=False):
    fake_pred = fake_pred.cpu().numpy()
    real_pred = real_pred.cpu().numpy()
    obs = obs.cpu().numpy()

    fake_traj = np.concatenate((obs, fake_pred), axis=0)
    real_traj = np.concatenate((obs, real_pred), axis=0)
    
    
    if all_three:
        x_obs, y_obs, x_pred, y_pred,num = generateTraj(fit_type='square', coeff=0.5, num=3) #changed by cuimingbo
        ax.plot(fake_pred[:,0], fake_pred[:,1], 'g', label='Predicted')
        ax.plot(x_obs[0,:],  y_obs[0,:],  'b',  label='Observed')
        ax.plot(x_pred[0,:], y_pred[0,:], 'r', label='Real Pred 1')
        if args.dataset_name != 'straight':
            ax.plot(x_pred[1,:], y_pred[1,:], 'r', label='Real Pred 2')
            if(num==3):
                ax.plot(x_pred[2,:], y_pred[2,:], 'r', label='Real Pred 3')
            else:
                pass
            # modified by cuimingbo
            if(num==5):
                ax.plot(x_pred[3,:], y_pred[3,:], 'r', label='Real Pred 4')
                ax.plot(x_pred[4,:], y_pred[4,:], 'r', label='Real Pred 5')
                # if args.dataset_name != 'fiveTraj' and args.dataset_name != 'uneqfiveTraj':
                    # ax.plot(x_pred[5,:], y_pred[5,:], 'r', label='Real Pred 4')
                    # ax.plot(x_pred[6,:], y_pred[6,:], 'r', label='Real Pred 5')

    else:
        # pass
        ax.plot(fake_pred[:,0], fake_pred[:,1], 'g')



def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d, teacher, mode
):
    with torch.no_grad():
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end) = batch
        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)

        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        if mode =='refinement':
            _,  traj_fake_rel = teacher.build_teacher(traj_fake_rel, traj_real_rel, seq_start_end)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()
    return losses


def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g, mode
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    if mode == 'training':
        optimizer_g.zero_grad()
        loss.backward()
        if args.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                generator.parameters(), args.clipping_threshold_g
            )
        optimizer_g.step()

    return losses


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

# def discriminator_refine_step(
#     args, batch, generator, discriminator, d_loss_fn, optimizer_d
# ):
#     batch = [tensor.cuda() for tensor in batch]
#     (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
#      loss_mask, seq_start_end) = batch
#     losses = {}
#     loss = torch.zeros(1).to(pred_traj_gt)

#     generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

#     pred_traj_fake_rel = generator_out
#     pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

#     traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
#     traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
#     traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
#     traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

#     ##Guide traj fake
#     _, self.optimal_batch = self.teacher.build_teacher(fake_batch, fake_sigmoid, fake_grad, real_batch)
#     ##LINK with Zero Grad
#     scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
#     scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

#     # Compute loss with optional gradient penalty
#     data_loss = d_loss_fn(scores_real, scores_fake)
#     losses['D_data_loss'] = data_loss.item()
#     loss += data_loss
#     losses['D_total_loss'] = loss.item()

#     optimizer_d.zero_grad()
#     loss.backward()
#     if args.clipping_threshold_d > 0:
#         nn.utils.clip_grad_norm_(discriminator.parameters(),
#                                  args.clipping_threshold_d)
#     optimizer_d.step()       

    # return losses
