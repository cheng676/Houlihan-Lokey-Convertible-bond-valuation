* TFmodel - ExplicitFDM - with call and put.ipynb: Original TF model pricing code passed from previous group, using Explicit FDM. Added call and put provision
* TFmodel - ExplicitFDM - Volatility calibration - with call and put.ipynb: Original TF model calibration code passed from previous group, using Explicit FDM, added call and put provision
* Result Analysis & Regression Adjustment on Model Price.ipynb:
* Functions.py: Definition of Crank-Nicolson method function for convergence analysis and synthetic data generation
* Crank-Nicolson - convergence and pricing.ipynb: Notebook for Crank-Nicolson method’s convergence analysis, calling functions.py


* Covergence_moneyness: This python notebook includes the code that need to be run on Nvidia GPU. It is a semi-product code that will lead to blow up the GPU memory. If anyone want to further work on minimize the numerical errors on TF models based on different moneyness, this code can be a good helper.
* CN_Mns_Cvg.ipynb: This Python notebook includes the code and functions that are used to solve TF model PDEs by Crank Nicolson method and test the convergence of these PDEs by different moneyness.
* Synthetic_data.ipynb: This Python notebook includes code to synthesize market data based on correlation matrices that are validated by statistical significance analysis.
* Cali_Vol_Rgrs.ipynb: This Python notebook includes code to perform general regression, log-transformed regression, and regression based on industry split, with Calibrated Volatility as target value and Time to Maturity, Moneyness, and Implied Volatilities as regressors.
* IVOL_TTM_Moneyness_Regressions.ipynb: This Python notebook includes code to perform regression, log-transformed regression, and regression based on industry split with Calibrated Volatility as target value, and time to maturity, moneyness, and implied volatilities as regressors. This is conducted on different Time to Maturity and Moneyness clusters of Convertible bond data and includes output scatter plots showing the results.
* Supervised Learning Models [RFR, GBR, MLP].ipynb: This Python notebook includes code to train random forest regression, XGBoost, and Neural Network models. The code takes as input a dataset, adds relevant features, performs a train test split, and then trains the supervised learning models, optimizing using grid search. The code also outputs a histogram of percentage errors, as well as a prediction vs test price plot.




* For RL code folder:

This has two folders; 


1.⁠ ⁠American Call options
2.⁠ ⁠⁠CB with all 3 algorithms
3.⁠ ⁠⁠Each folder has it’s own README.md to run


It has enough of comments in each of the file.
1. American-Call-Option Pricing (QuantLib + Static NN + RL)
File
	What it does
	quantlib_data_generation.py
	Calls QuantLib to create a CSV of ground-truth American-call prices from input parameters [S, K, r, q, σ, T].
	main_static_american_call.py
	Reads the CSV and trains a static feed-forward network that regresses the option price; saves static_model.pth.
	american_call_env.py
	Gym-style environment in which an agent decides each day whether to “hold” or “exercise” the option.
	rl_agent.py
	REINFORCE-based policy-gradient agent used by the dynamic RL approach.
	main_american_call_rl.py
	Runs many simulated price paths, updates the RL agent’s policy, and writes rl_policy_model.pth.
	eval_american_call.py
	Loads both models and the QuantLib prices, prints RMSEs, and plots price-vs-truth & exercise boundary figures.
	(Generated) data/american_call_data.csv
	Output of quantlib_data_generation.py; used as the single source of truth for both training tracks.
	requirements.txt
	Pins QuantLib-Python, PyTorch, pandas, matplotlib, etc., so the experiments reproduce exactly.
	

Convertible-Bond Pricing - Reinforcement Learning
Core, high-level files (as described in the project README)
File
	What it does
	envs.py
	Two Gym environments: DiscretePricePredictionEnv & ContinuousPricePredictionEnv.
	agents.py
	Three independent learners – Q-Learning, vanilla REINFORCE, and DDPG – all wrapped in clean classes.
	utils.py
	Shared helpers: data loaders, plotting utilities, bucket/action converters, etc.
	training.py
	Outer training loops that glue each algorithm to an environment and handle checkpoints.
	main.py
	CLI entry point; parses `--mode {q_learning/ddpg/policygrad}