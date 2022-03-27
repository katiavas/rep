import numpy as np
import torch as T
from actor_critic import ActorCritic
from icm import ICM
from memory import Memory
from utils import plot_learning_curve
from wrappers import make_atari
import random
from utils import plot_learning_curve_with_shaded_error
# import wandb


def worker(name, input_shape, n_actions, global_agent,
           optimizer, env_id, n_threads, global_idx, global_icm,
           icm_optimizer, icm):
    T_MAX = 20

    local_agent = ActorCritic(input_shape, n_actions)

    if icm:
        local_icm = ICM(input_shape, n_actions)
    else:
        local_icm = None
        intrinsic_reward = None

    memory = Memory()

    frame_buffer = [input_shape[1], input_shape[2], 1]
    env = make_atari(env_id, shape=frame_buffer)

    env.seed(111)


    episode, max_steps, t_steps, scores = 0, 5000, 0, []
    intr = []
    l = []
    l_i = []
    l_f = []

    while episode < max_steps:
        obs = env.reset()
        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)
        while not done:
            state = T.tensor([obs], dtype=T.float)
            action, value, log_prob, hx = local_agent(state, hx)
            obs_, reward, done, info = env.step(action)

            memory.remember(obs, action, obs_, reward, value, log_prob)
            score += reward
            obs = obs_
            ep_steps += 1
            t_steps += 1
            if ep_steps % T_MAX == 0 or done:
                states, actions, new_states, rewards, values, log_probs = \
                        memory.sample_memory()
                if icm:
                    intrinsic_reward, L_I, L_F = \
                            local_icm.calc_loss(states, new_states, actions)
                    # wandb.log({'forward_loss':L_F.item(), 'inverse_loss':L_I.item(), 'intrinsic_reward': intrinsic_reward})
                loss = local_agent.calc_loss(obs, hx, done, rewards,
                                             values, log_probs,
                                             intrinsic_reward)
                optimizer.zero_grad()
                hx = hx.detach_()
                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()
                loss.backward()
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)
                for local_param, global_param in zip(
                                        local_agent.parameters(),
                                        global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(
                                        local_icm.parameters(),
                                        global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())

                memory.clear_memory()
        episode += 1
        # wandb.log({'episode_score': score})
        # with global_idx.get_lock():
        #    global_idx.value += 1
        if name == '1':
            loss_i = T.sum(L_I)
            l_i.append(loss_i)
            loss_f = T.sum(L_F)
            l_f.append(loss_f)
            b = T.sum(loss)
            l.append(b.detach().numpy())
            a = T.sum(intrinsic_reward)
            intr.append(a.detach().numpy())  # for plotting intrinsic reward
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_score_5000 = np.mean(scores[max(0, episode-5000): episode+1])
            print('ICM episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'avg score (100) {:.2f}'.format(
                                                episode, name, n_threads,
                                                t_steps/1e6, score,
                                                avg_score))
    if name == '1':
        x = [z for z in range(episode)]
        # plot_learning_curve(x, scores, 'Cartpole_pixels_ICM.png')
        np.savetxt("Breakout_111_ICM.csv",
                   scores,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("Breakout_111_ICM_intr.csv",
                   intr,
                   delimiter=",",
                   fmt='% s')

        np.savetxt("ICM_ON_LOSS_111.csv",
                   l,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("L_I_111.csv",
                   l_i,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("L_F_111.csv",
                   l_f,
                   delimiter=",",
                   fmt='% s')
        # plot_learning_curve_with_shaded_error(x, scores, 'ICM_shaded_error_5000.png')
