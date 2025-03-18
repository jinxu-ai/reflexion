import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

print(f"------------------------------------------------ here is the new round. [for new record] -------------------------------------------------")


# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = get_environment(env_type)(config, train_eval='eval_out_of_distribution')
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()

print( f"the observation: \n {obs} \n\n")

print(f"the infomation: \n {info} \n\n")


obs = '\n'.join(obs[0].split('\n\n')[1:])

print(f"\n the project obs: \n{obs}\n\n")

name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
print(f"\n the project name: \n{name}\n\n")

i = 0
while i < 20:
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    print(f"admissible_commands: {admissible_commands}")
    print(type(admissible_commands))
    print(len(admissible_commands))
    print(admissible_commands[0])
    print(type(admissible_commands[0]))
    print(len(admissible_commands[0]))
    random_actions = [np.random.choice(admissible_commands[0])]

    # step
    obs, scores, dones, infos = env.step(random_actions)
    print(f"obs: {obs} \n scores: {scores} \n dones: {dones} \n infos: {infos} \n \n")

    print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))
    i += 1
    print(f"\n -------------the {i}-th round---------- \n")