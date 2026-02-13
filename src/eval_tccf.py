import os
import sys
import subprocess
from termcolor import cprint
from omegaconf import DictConfig, ListConfig, OmegaConf

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":
    config = get_config()

    project_name = config.experiment.project
    eval_type = config.dataset.data_type

    def build_cli_args():
        cli_args = []
        
        for arg in sys.argv[1:]:
            if not arg.startswith('config='):
                cli_args.append(arg)
        
        return ' '.join(cli_args)

    cli_params = build_cli_args()
    
    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
        
    def sample(model_base):
        cprint(f"This is sampling.", color = "green")
        
        base_cmd = f'python {model_base}_sample_tccf.py'
        
        if cli_params:
            full_cmd = f'{base_cmd} {cli_params}'
        else:
            full_cmd = base_cmd
          
        subprocess.run(
            full_cmd,
            shell=True,
            cwd='sample',
            check=True,
        )
    
    def reward():
        cprint(f"This is the rewarding.", color = "green")
        
        base_cmd = f'python reward.py'
        full_cmd = f'{base_cmd} {cli_params}' if cli_params else base_cmd
        
        subprocess.run(
            full_cmd,
            shell=True,
            cwd='reward',
            check=True,
        )
    
    def execute():
        cprint(f"This is the execution.", color = "green")
        
        base_cmd = f'python execute.py'
        full_cmd = f'{base_cmd} {cli_params}' if cli_params else base_cmd
        
        subprocess.run(
            full_cmd,
            shell=True,
            cwd='reward',
            check=True,
        )
    
    os.makedirs(config.paths.result_base, exist_ok=True)
    
    sample(config.model_base)
    if eval_type == "code":
        execute()
    
    reward()
