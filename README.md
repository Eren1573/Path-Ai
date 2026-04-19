# 🚗 PATH AI – Autonomous Driving Simulation using Reinforcement Learning

PATH AI is a reinforcement learning-based autonomous driving simulation that demonstrates intelligent decision-making in a dynamic multi-lane highway environment. The system uses simulated LIDAR sensing to perceive surroundings and a Deep Q-Network (DQN) to learn safe driving behavior such as lane switching, obstacle avoidance, and collision prevention.

🎯 Key Features
🧠 Reinforcement Learning (DQN-based decision making)
📡 Simulated LIDAR sensor using ray-casting
🛣️ 3-lane highway environment
🚗 Multiple dynamic traffic vehicles
🐄 Animals crossing the road (random events)
🔄 Smooth lane-switching mechanism
📊 Real-time HUD:
Speed display
Collision counter
Distance travelled
🎮 Real-time simulation using Pygame
🧠 How It Works

The system follows a complete autonomous driving pipeline:

Environment → LIDAR → State → DQN Agent → Action → Environment
LIDAR scans surroundings
Distance values form the state
DQN predicts best action
Car performs lane change / stay
Reward is given based on outcome
Model learns over time
🛠️ Tech Stack
Python 🐍
Pygame (Simulation & Visualization)
PyTorch (Deep Learning)
NumPy
🚀 Installation
git clone https://github.com/your-username/path-ai.git
cd path-ai
pip install pygame torch numpy
python path_ai_full_simulation.py
📸 Output

The simulation includes:

Real-time vehicle movement
LIDAR rays visualization
Dynamic traffic and obstacles
Autonomous lane decision making
📊 Metrics Displayed
Speed
Collision count
Distance travelled
🧪 Learning Approach
Deep Q-Network (DQN)
Experience replay
Epsilon-greedy exploration
Reward-based learning
⚖️ Advantages
Adaptive learning system (not rule-based)
Dynamic and unpredictable environment
Real-time visualization of AI decisions
Scalable architecture
⚠️ Limitations
Simplified physics (no real vehicle dynamics)
Simulated sensors (no real hardware)
Limited to 2D environment
🔮 Future Scope
Multi-sensor fusion (LIDAR + Camera + Radar)
Integration with simulators like CARLA Simulator
Realistic vehicle physics and weather conditions
Advanced RL algorithms (PPO, SAC)
Traffic signals and urban driving scenarios
Model saving/loading and evaluation mode
🧩 Project Structure

path-ai/
│
├── path_ai_full_simulation.py
├── README.md

💡 Unique Aspects
End-to-end autonomous pipeline (Perception → Decision → Action)
Reinforcement learning instead of rule-based logic
Real-time LIDAR-based state representation
Multi-lane decision-making system
🎓 Use Cases
Autonomous driving research
AI/ML academic projects
Simulation-based learning
Reinforcement learning experimentation

👨‍💻 Author
B. Sai Naga Sowri
Software Developer | AI/ML Enthusiast

⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork it
📢 Share it
