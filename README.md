# Forecasting Frontier Language Model Agent Capabilities

This repository contains all coded needed to reproduce our paper and website.
The website is live at [https://pimpale.github.io/forecasting-frontier-language-model-agent-capabilities/](https://pimpale.github.io/forecasting-frontier-language-model-agent-capabilities/).

### Paper Plots
All the code needed to reproduce the plots in the paper is included in the `./graphs/` directory.
It's kind of messy, but the key files that you need to be aware of are:
* Data:
    * `./graphs/data_models/cache_new/agentic_benchmark.csv`: this is the agentic data we use for our forecasts
    * `./graphs/data_models/meta/openllm_elo_merged.csv`: this is the data we use for backtesting
* Code:
    * `./graphs/backtesting_frontier.py`: this is the code we use to both backtest and make forecasts
    * `./graphs/backtesting_metrics.py`: this is the code we use to backtest the metrics.
    * `./graphs/util*`: these files are utility files that are used by the other files.
    * `./graphs/datagen*`: these files are used to generate the data we use for the forecasts.

### Baseline Agent
The baseline agent used in the experiments is included in `./unelicited_agent/`.
It uses the Inspect framework to run the agent in a simulated environment. 
There are scripts for running the agent in the `./unelicited_agent/` directory. 

### Website
The code for the website is included in `./website/`.
It's a standard React + vite project. Further instructions can be found in the `./website/` directory. 