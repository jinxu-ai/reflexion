"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import importlib
import alfworld
import alfworld.agents.environment

# import openai API - completion & chat 
from utils import Model, get_chat, get_completion
# ollama generate & chat function
from redo_test.chat_test import ollama_chat, ollama_generate

from env_history import EnvironmentHistory

from typing import List, Dict, Any, Tuple
 

 # obtain the openai key - 
 # local host - not API needed
openai.api_key = os.environ["OPENAI_API_KEY"]

# prompt file path - json file
FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'

# open the promp json file
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

'''
function: llm
parameter:
    prompt - string 
    model - model function
    stop - when the string encounts a '\n' symbol, stops

'''
def llm(prompt: str, model: Model, stop: List[str] = ["\n"]):
    try:
        cur_try = 0
        # set the maximum try time is 6
        while cur_try < 6:
            # use the model: "text-davinci-003" - GPT3.5
            # ollama model: "llama3.2"
            if model == "llama3.2":
                # Call get_completion function - generate text 
                text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
                # text = ollama_generate("llama3.2", prompt)
            else:
                # Call get_chat function - having a chat
                text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
                # text = ollama_chat("llama3.2", prompt)
            # dumb way to do this
            if len(text.strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        # string - prompt
        print(prompt) 
        # exception
        print(e)
        import sys
        sys.exit(1)

'''
ob(observation)
check the current location
'''
def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


'''
alfworld run function
parameters: 
    env
    base_prompt
    memory - list: maximum length is 3
    to_print
    ob - observation: initial value=''
    model - change to ollama 'llama3.2'

return: Tuple[EnvironmentHistory, bool]

object: EnvironmentHistory (from env_history file, class EnvironmentHistory)

'''
def alfworld_run(env, base_prompt, memory: List[str], to_print=True, ob='', model: Model = "llama3.2") -> Tuple[EnvironmentHistory, bool]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()

    # whether to print the observation 
    if to_print:
        print(ob)
        sys.stdout.flush()

    # current step count
    cur_step = 0

    # max step = 49, for one alfworld problem
    while cur_step < 49:
        # action is chosen by LLM("llama3.2")
        # .strip() - string function: remove the start and end of the string or specific character
        action = llm(str(env_history) + ">", stop=['\n'], model=model).strip()

        # record the environment history
        env_history.add("action", action)

        # observation, reward, done, information - from the environment's step variable 
        # environment step?
        # reward is not defined (not used)
        observation, reward, done, info = env.step([action])

        # observation: process_ob
        # reward: info['won'][0]
        # done: done[0]
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]


        if action.startswith('think:'):
            observation = 'OK.'
        
        # record the observation 
        env_history.add("observation", observation)



        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        
        # flag to stop the while loop
        if done:
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        
        # count the step
        cur_step += 1
    # default return
    return env_history, False

# prefixes
PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}



'''
trial 

parameters:
    trial_log_path - file path?
    world_log_path
    trial_idx - id of the trial
    env_configs - List[Dict[str, Any]]: ?

    use_memory - True/False
    model - "llama3.2"

return: 
    List[Dict[str, Any]]

'''
def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model: Model,
    ) -> List[Dict[str, Any]]:

    # reload module 'alfworld' in this trial 
    importlib.reload(alfworld)
    # reload sub module 'alfworld.agents.environment' in this trial
    importlib.reload(alfworld.agents.environment)

    # open yaml file to load the basic configuration
    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)

    # split? 
    split = "eval_out_of_distribution"

    # 获取 alfworld.agents.environment 模块中，名称为 config["env"]["type"] 的属性
    # 在获取到的属性后面加上括号 (config, train_eval=split)，表示调用该属性
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    # Iterate through the list of environment configurations provided by `env_configs`
    for z, env_config in enumerate(env_configs):
        # Reset the environment and retrieve the initial observation and info
        ob, info = env.reset()

        # Process the initial observation text, keeping only the content from the second section onwards
        ob = '\n'.join(ob[0].split('\n\n')[1:])

        # Extract the environment name from the path in the info (taking the last three and second-to-last directories)
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        # Print the current environment name being used
        print(f"using {name}")

        # Check if the current environment has already been successfully solved
        if env_config["is_success"]:
            num_successes += 1

            # Log the success to the world log file
            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            
            # Log the success to the trial log file
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            
            # Skip further processing for this environment and continue to the next configuration
            continue

        # Iterate through the prefixes in the dictionary `PREFIXES` to find one that matches the environment name
        for i, (k, v) in enumerate(PREFIXES.items()):
            # If the environment name starts with the prefix `k`
            if name.startswith(k):
                # Construct a base prompt using two example tasks based on the prefix
                base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
                
                # Run the environment using `alfworld_run`
                # Parameters include the environment, the prompt, memory (if `use_memory` is True), 
                # whether to print output, the initial observation, and the model
                final_env_history, is_success = alfworld_run(env, base_prompt, env_config["memory"] if use_memory else [], to_print=True, ob=ob, model=model)

                # Update the environment configuration status
                # update env config
                if is_success:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    env_configs[z]['is_success'] = True  # Mark the environment as successfully solved
                    num_successes += 1  # Increment total success count
                    num_additional_successes += 1 # Increment additional success count
                else:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'


                # Log the current environment status to the world log file
                # log to world log
                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')
                
                
                # Log the final environment history and status to the trial log file
                # log env results to trial log
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    # close environment object
    env.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs
