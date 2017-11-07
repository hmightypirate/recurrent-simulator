# recurrent-simulator


Implementation in Chainer and ChainerRL of a simplified version of the paper Recurrent Environment Simulators from Google Deepmind (https://arxiv.org/abs/1704.02254)


This is a simple toy example. Hence many differences exists with the original paper, such as

- Training/Test methodology. Here an agent plays continuously generating new samples.
- Samples are stored in memory, but no BPTT is applied to train with longer sequences.
- Images are downscaled and transformed to gray images before entering the simulator.


The agent that generates samples is an A3C (https://github.com/hmightypirate/guided-backprop-chainerrl) but any other ChainerRL agent could be easily integrated with this simulator, simply by modifying the training script "examples/mygym/train_a3c_gym.py". To avoid training the agent whilst the samples for the recurrent environment are generated a dummy_a3c_agent is employed (that only acts but never trains).

## Dependencies

- gym and gym atari: pip install gym gym['atari']
- chainer: pip install chainer
- chainerrl (https://github.com/chainer/chainerrl). Download and install with pip install . --upgrade

## Training

The sample script training_script_breakout.sh trains a Breakout environment saving it in the "break" folder (Waring: the folder in which the model is stored should exist)

```
OMP_NUM_THREADS=1 python examples/mygym/train_a3c_gym.py 1 --env Breakout-v0 --outdir outdir --t-max 5 --lr 7e-4 --min_reward -500 --beta 1e-2 --reward-scale-factor 1.0 --logger-level 20 --rmsprop-epsilon 1e-1 --eval-interval 10000000 --eval-n-runs 0 --savesim break --warm_up_steps 600  --saving_steps 3000  --obs_dep 10 --pred_dep 15 --num_output_channels_sim 512 --batch_size 16

```

## Testing

```
OMP_NUM_THREADS=1 python examples/mygym/train_a3c_gym.py 1 --env Breakout-v0 --outdir outdir --t-max 5 --lr 7e-4 --min_reward -500 --beta 1e-2 --reward-scale-factor 1.0 --logger-level 20 --rmsprop-epsilon 1e-1 --eval-interval 10000000 --eval-n-runs 10 --loadtosim break --demo --render-b2w --chain_period 100 --num_output_channels_sim 512
```

## Using a trained agent

To use a trained agent add the --load <folder with the model> parameter

## Reloading from checkpoint
To continue the training from a checkpoint add the --loadtosim <folder with the env model> to the parameter list. This parameter is also used during testing to load the environment.


## Files

* env_sim_chainer: this is the simulator environment. Wraps the original environment and predicts new environment states as a function of the actions of the agent and the previous states of the environment (real or predicted)* iclr_acer_link: implements the convolutional/deconvolutional networks of angent and environment.
* recurrent_env: implements the training/testing methods. The step function also adds the environment state (image + agent action) to the agent buffer.
* guieded_relu: a relu with guided backpropagation (for visualization of the saliency maps in the agent)
* env_gym_chainer: a wrapper of the gym environment (useful for training agents with states composed of several frames)
* cond_lstm: a slightly modified lstm to calculate action/state interactions
* agent_data_set: the buffer to store samples to train the environment simulator
* saliency_a3c: an A3C agent that also calculates the saliency maps during testing
* recurrent: a modified version fo the recurrent file in chainerRL (because we are using here a modified LSTM, not the LSTM itself).
* saliency_dummy_a3c: a dummy a3c agent that plays but never trains.
* train_a3c_gym: the main script that sets up the whole system. 

# Parameters

To obtain a complete description of input parameters just type:

```
python examples/mygym/train_a3c_gym.py -h
```
