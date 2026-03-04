# HCRide

**HCRide: Harmonizing Passenger Fairness and Driver Preference for Human-Centered Ride-Hailing**  
Code and data for the IJCAI-25 paper by Lin Jiang, Yu Yang, and Guang Wang.



## 📌 Overview

HCRide is a two-layer decision framework designed to optimize ride-hailing services by balancing passenger fairness and driver preference. This repository includes the full pipeline: dispatch algorithms, simulation environment, and experiment scripts.


## 📁 Project Structure

```
├── algorithm/       # Core dispatch and sensing algorithms
├── data/            # Input datasets and preprocessing scripts
├── run/             # Entry scripts for launching experiments
├── simulator/       # Simulation environment implementation
├── test/            # Testing and evaluation scripts
└── README.md        # Project documentation
```


## 🚀 Getting Started

### 1. Clone the Repository

```bash
conda create --name HCRide python=3.7
git clone https://github.com/your-org/HCRide.git
cd HCRide
```

### 2. Install Dependencies

```bash
conda activate HCRide
pip install -r requirement.txt
```

### 3. Run the Main Pipeline

```bash
python run/run.py
```
The trained agent will be saved in the `result/` directory.

## 🧪 Running Tests

You can run the evaluation script via:

```bash
python test/execute.py
```

Results will be saved to the `result/` folder.


## 📖 Citation

If you use this code or dataset, please cite the following paper:

```bibtex
@inproceedings{jiang2025hcride,
  title={HCRide: Harmonizing Passenger Fairness and Driver Preference for Human-Centered Ride-Hailing},
  author={Jiang, Lin and Yang, Yu and Wang, Guang},
  booktitle={Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}
```


*This project is part of the IJCAI-25 Special Track on Human-Centered Artificial Intelligence.*
