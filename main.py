import torch.nn as nn
from pde.FreeBoundaryPDE import BSData
from trainer.pde_trainer import PDETrainer
from pde.utils.payoffs.arithmetic_payoff import ArithmeticPayoff

if __name__ == '__main__':
    parameters_dict = {
        'n_epochs': 60000,
        'input_dim': 2,
        'output_dim': 1,
        'n_layers': 3,
        'n_units': 100,
        'sampler_data': {
            "n_points": 5000,
            "time_dim": True,
            "space_dim": 1,
            "time_range": [0, 1],
            "space_range": [0, 1]
        },
        'pde_data': {
            "time_dim": True,
            "bs_data": BSData(
                interest_rate=0,
                volatility=0.75,
                correlation=0.5,
                dividend_rate=0,
                strike_price=0.5,
                payoff=ArithmeticPayoff()
            )
        },
        'activation_fn': nn.Tanh(),
        'output_activation_fn': None,
        'skip_connection': False,
        'init_method': nn.init.xavier_uniform,
        'criterion': None,
        'optimizer': None,
        'learning_rate': 3e-5,
        'learning_rate_scheduler': "ema",
        'learning_rate_scheduler_params': {
            "gamma": 0.99
        },
        'device': "mps"
    }

    trainer = PDETrainer(**parameters_dict)
    trainer.run()
