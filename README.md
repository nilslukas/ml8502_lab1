# ML8502 @MBZUAI - Lab 1
## Instructor: Nils Lukas 

This repository contains the code and instructions for Lab 1 of the ML8502 course at MBZUAI. The lab focuses on (1) Data Poisoning Defenses (Neural Cleanse) and (2) Watermarking in Text Generation (KGW).

This readme serves as a simple setup guide to get you started with the lab exercises. Follow the instructions below to set up your environment and run the provided code.

## Setup Instructions
> Note: If you do not want to use uv, feel free to use your prefered package manager, and install dependencies from `requirements.txt`.
### Prerequisites
We will use uv for environment management. If you don't have it installed, you can find the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

### Step 1: Clone the Repository
First, clone this repository to your local machine using the following command:
```bash
git clone git@github.com:nilslukas/ml8502_lab1.git
cd ml8502_lab1
```
### Step 2: Create and Activate the Virtual Environment
Create a new virtual environment using uv:
```bash
uv init
```
### Step 3: Install Required Packages
Install the required Python packages using pip:
```bash
uv add -r requirements.txt
```
### Step 4: Run the Lab Code
You can now run the lab code. For example, to run the data poisoning attack script,
use the following command:
```bash
uv run neural_cleanse_exercise.py
```
Similarly, to run the watermarking script, use:
```bash
uv run kgw_watermark_exercise.py
```

Good luck with your lab exercises! If you have any questions, feel free to reach out to the instructor or your peers.