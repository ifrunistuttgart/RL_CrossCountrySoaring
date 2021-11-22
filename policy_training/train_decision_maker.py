import datetime
import os
import shutil
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections

from policy_training.decision_maker import evaluate_decision_maker, params_decision_maker
from policy_training.decision_maker.ppo_decision_maker import PPO
from policy_training.parameters import params_triangle_soaring, params_environment
from policy_training.subtasks.updraft_exploiter import model_updraft_exploiter
from policy_training.subtasks.vertex_tracker.waypoint_controller import ControllerWrapper

# Choose device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def create_experiment_folder(_params_rl, _params_agent, _params_logging):

    # create folder to store data for the experiment running
    experimentID = 1
    dirName = "{}_decision_maker_experiment".format(experimentID)
    while os.path.exists(dirName):
        experimentID += 1
        dirName = "{}_".format(experimentID) + dirName.split('_', 1)[1]
    os.mkdir(dirName)
    shutil.copytree(os.getcwd(), os.path.join(dirName, "Sources_unzipped"),
                    ignore=shutil.ignore_patterns('*experiment*', 'archive', 'tests', '.git*', '../rl_ccs_experiments',
                                                  '.idea', '__pycache__', 'README*'))
    os.chdir(dirName)
    shutil.make_archive("Sources", 'zip', "Sources_unzipped")
    shutil.rmtree("Sources_unzipped")
    print("Directory for running experiment no. {} created".format(experimentID))

    # save parameters to file
    parameterFile = open("parameterFile.txt", "w")
    parameterFile.write(
        format(vars(_params_rl)) + "\n" +
        format(vars(params_decision_maker.ModelParameters())) + "\n" +
        format(vars(_params_agent)) + "\n" +
        format(vars(_params_logging)) + "\n\n" +
        format(vars(params_triangle_soaring.TaskParameters())) + "\n\n" +
        format(vars(params_environment.SimulationParameters())) + "\n" +
        format(vars(params_environment.GliderParameters())) + "\n" +
        format(vars(params_environment.PhysicsParameters())) + "\n" +
        format(vars(params_environment.WindParameters())))
    parameterFile.close()

    # set up file to save average returns and scores
    returnFile = open("returnFile_running.dat", "w")
    returnFile.write("iterations,episodes,avg_returns,avg_scores\n")
    returnFile.close()

    return parameterFile, returnFile


def run_decision_maker_training():
    """ Controls training process of the decision maker

    """
    # set up training
    env = gym.make('glider3D-v0', agent='decision_maker')

    # instantiate vertex-tracker and updraft-exploiter
    waypoint_controller = ControllerWrapper(env)
    updraft_exploiter = model_updraft_exploiter.UpdraftExploiterActorCritic().to(device)
    updraft_exploiter.load_state_dict(torch.load("updraft_exploiter_actor_critic_final_17-December-2020_11-06.pt"))

    # instantiate agent
    ppo = PPO(waypoint_controller, updraft_exploiter, env)

    # load parameters
    _params_rl = params_decision_maker.LearningParameters()
    _params_agent = params_decision_maker.AgentParameters()
    _params_logging = params_decision_maker.LoggingParameters()

    parameterFile, returnFile = create_experiment_folder(_params_rl, _params_agent, _params_logging)

    # set random seed
    if _params_rl.SEED:
        print("Random Seed: {}".format(_params_rl.SEED))
        torch.manual_seed(_params_rl.SEED)
        env.seed(_params_rl.SEED)
        np.random.seed(_params_rl.SEED)

    # initialize logging variables
    returns = collections.deque(maxlen=10)
    scores = collections.deque(maxlen=10)
    average_returns = []
    average_scores = []
    policy_iterations = 0
    episodes = 0
    interactions = 0
    ret = 0

    # create sample of untrained system behavior
    evaluate_decision_maker.main(env, ppo, policy_iterations, _params_agent, validation_mask=True)

    # (re-)set env and model
    env.reset()
    obs = env.get_observation()
    lstm_in = ppo.model.reset_lstm()

    # training loop
    while policy_iterations < (int(_params_rl.N_ITERATIONS)):

        # rollout s.t. current policy
        with torch.no_grad():
            action_env, action_agent, action_agent_logprob, state_value, lstm_out = \
                ppo.select_action(torch.FloatTensor(obs), lstm_in)
            next_obs, reward, done, _ = env.step(action_env)

        # store data in ppo.buffer
        ppo.buffer.store(obs=torch.FloatTensor(obs).to(device), act=action_agent.flatten(),
                         rew=torch.FloatTensor([reward / 100]), val=torch.FloatTensor([state_value]),
                         logp=action_agent_logprob.flatten(), lstm_h_in=lstm_in[0].flatten(),
                         lstm_c_in=lstm_in[1].flatten(), done=torch.FloatTensor([done]))

        # update variables each interaction
        obs = next_obs
        lstm_in = lstm_out
        ret += reward
        interactions += 1

        # store results and reset experiment if episode is completed
        if done:
            ppo.buffer.finish_path(torch.FloatTensor([0]).to(device))
            returns.append(ret)
            scores.append(env.lap_counter * 200)
            episodes += 1
            ret = 0
            env.reset()
            lstm_in = ppo.model.reset_lstm()
            obs = env.get_observation()

            n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
            average_returns.append(np.convolve(list(returns)[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])
            average_scores.append(np.convolve(list(scores)[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        # update policy every BATCHSIZE interactions
        if interactions != 0 and interactions % _params_rl.BATCHSIZE == 0:
            next_value = ppo.model.critic(torch.FloatTensor(obs)).detach()
            ppo.buffer.finish_path(next_value.flatten())
            ppo.model.train()
            ppo.update()
            ppo.model.eval()
            interactions = 0
            policy_iterations += 1

            # print results to display and log them to file
            if len(average_returns) and policy_iterations % _params_logging.PRINT_INTERVAL == 0:
                print("# policy iteration: {}/{} \t\t"
                      "avg. return over last 10 episodes: {:06.1f} \t\t"
                      "avg. score over last 10 episodes: {:06.1f}"
                      .format(policy_iterations, int(_params_rl.N_ITERATIONS), average_returns[-1], average_scores[-1]))

                with open("returnFile_running.dat", "a+") as return_file:
                    return_file.write(format(policy_iterations) + "," + format(episodes) + "," +
                                      '{:.1f}'.format(average_returns[-1]) + "," + '{:.1f}'.format(average_scores[-1]) +
                                      "\n")

            # save model and create sample of system behavior
            if policy_iterations % _params_logging.SAVE_INTERVAL == 0:
                torch.save(ppo.model.actor.state_dict(),
                           "decision_maker_actor_iter_{}".format(policy_iterations) + ".pt")
                torch.save(ppo.model.critic.state_dict(),
                           "decision_maker_critic_iter_{}".format(policy_iterations) + ".pt")
                evaluate_decision_maker.main(env, ppo, policy_iterations, _params_agent, validation_mask=True)

                fig, ax = plt.subplots()
                returns_to_plot = pd.read_csv('returnFile_running.dat')
                returns_to_plot.plot(x='iterations', y='avg_returns', ax=ax)
                returns_to_plot.plot(x='iterations', y='avg_scores', ax=ax)
                plt.title("learning success")
                plt.xlabel("policy iterations (-)")
                plt.ylabel("average return/score (-)")
                plt.grid(True)
                plt.savefig("average_returns_{}".format(policy_iterations) + ".png")
                plt.show()

    # save final model
    now = datetime.datetime.now()
    torch.save(ppo.model.actor.state_dict(), "decision_maker_actor_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")
    torch.save(ppo.model.critic.state_dict(), "decision_maker_critic_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    # rename parameter file consistently
    os.rename(parameterFile.name, "parameters_" + now.strftime("%d-%B-%Y_%H-%M") + ".txt")

    # rename return file consistently
    return_file.close()
    os.rename(return_file.name, "average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".dat")

    env.close()


if __name__ == '__main__':
    run_decision_maker_training()
