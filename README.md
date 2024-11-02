# FANETsRounting: A Network Simulation with Reinforcement Learning

## Project Overview
This project implements a network simulation environment that utilizes reinforcement learning techniques, specifically Q-learning, to manage routing and energy consumption among nodes in a simulated network. The simulation explores how nodes can efficiently communicate and adapt based on their local environment and the overall network state.

## Environment and Network Components
### Network Class
The `Network` class manages the overall network simulation, including node initialization, packet handling, and the communication process. It supports various reinforcement learning agents for routing decisions.

### Node Class
The `Node` class represents each node in the network. It is responsible for:
- Sending and receiving packets.
- Managing its energy consumption.
- Participating in the routing decision-making process.

## Reinforcement Learning in Routing
The project implements three types of reinforcement learning agents:
1. **Q-learning**: Utilizes a Q-table to learn optimal routing paths.
2. **DQN**: Deep Q-Learning, which employs neural networks for action selection.
3. **DDPG**: Deep Deterministic Policy Gradient, used for continuous action spaces.

### QLearningRouting Class
The `QLearningRouting` class is central to the Q-learning approach, handling:
- State representation based on the destination node.
- Action selection using an epsilon-greedy policy.
- Q-table updates based on received rewards from the environment.

## Simulation Flow
The main simulation loop is managed by the `Network.simulate()` method, which oversees:
- Node routines for sensing and communication.
- Clustering and neighbor discovery processes.
- Energy calculations and performance metrics.

## Configuration and Running the Simulation
Configuration parameters can be set in the `config.py` file, including:
- Number of nodes.
- Area dimensions.
- Reinforcement learning parameters.

To run the simulation, ensure all dependencies are installed, then execute the main script. Results and logs will be generated for analysis.

## Output and Results
The simulation generates logs and output files detailing:
- Energy consumption metrics.
- Packet delivery rates.
- Performance data for the Q-learning agent.

Results can be found in the `RL_Results` directory after running the simulation.

## Future Improvements
Potential areas for improvement include:
- Experimenting with different reinforcement learning algorithms.
- Enhancing node mobility models.
- Integrating more sophisticated energy consumption models.

