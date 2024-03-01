# OfflineArcher
Research Code for the Offline Experiments of "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL" 

[Yifei Zhou](https://yifeizhou02.github.io/), [Andrea Zanette](https://azanette.com/), [Jiayi Pan](https://www.jiayipan.me/), [Aviral Kumar](https://aviralkumar2907.github.io/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)

![archer_diagram 001](https://github.com/YifeiZhou02/ArCHer/assets/83000332/b874432a-d330-49a5-906c-bba37e17f831)


This repo supports the following methods:

- [Offline ArCHer][1]
- Offline Filtered BC
- Offline BC

[1]: https://github.com/YifeiZhou02/ArCHer

And the following environments
- [Twenty Questions][2]

[2]: https://lmrl-gym.github.io/


## Quick Start
### 1. Install Dependencies
```bash
conda create -n archer python==3.10
conda activate archer

git clone https://github.com/andreazanette/OfflineArcher
cd OfflineArcher
python -m pip install -e .
```
### 2. Download Datasets and Oracles
Offline datasets and Oracles checkpoints used in the paper can be found [here](https://drive.google.com/drive/folders/1pRocQI0Jv479G4vNMtQn1JOq8Shf2B6U?usp=sharing).
You will need to create an "oracles" and "datasets" folder and put the oracle and dataset in such folders.
The oracle for Twenty Questions should be named 20q_t5_oracle.pt and the dataset should be called "twenty_questions.json".

### 3. Run Experiments
You can directly run experiments by runnig the launch scripts. For example, in order to lauch Offline Archer on Twenty Question simply run
```bash
. submit_OfflineArcher_TwentyQuestions.sh
```
The code uses the torch lightning framework. Please refer to the documentation of torch lightning (https://lightning.ai/docs/pytorch/stable/) for additional information, such as using different flags when launching the code.

### 4. Citing Archer
```
@misc{zhou2024archer,
      title={ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL}, 
      author={Yifei Zhou and Andrea Zanette and Jiayi Pan and Sergey Levine and Aviral Kumar},
      year={2024},
      eprint={2402.19446},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

