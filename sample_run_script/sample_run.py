from chronoclust.main import main


data_directory = '../synthetic_dataset'
data_files = ['{}/synthetic_d{}.csv.gz'.format(data_directory, x) for x in range(5)]

config = {
    "beta": 0.2,
    "delta": 0.05,
    "epsilon": 0.03,
    "lambda": 2,
    "k": 4,
    "mu": 0.01,
    "pi": 3,
    "omicron": 0.00000435,
    "upsilon": 6.5
}

output_directory = '../sample_run_script/output'

main.run(data=data_files,
         output_directory=output_directory,
         param_beta=config['beta'],
         param_delta=config['delta'],
         param_epsilon=config['epsilon'],
         param_lambda=config['lambda'],
         param_k=config['k'],
         param_mu=config['mu'],
         param_pi=config['pi'],
         param_omicron=config['omicron'],
         param_upsilon=config['upsilon']
         )

