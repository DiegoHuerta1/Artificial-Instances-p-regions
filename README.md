# Artificial-Instances-p-regions


This repository contains a Python library to **generate artificial instances for the p-regions problem**.  
It includes ready-to-use scripts and examples so you can quickly create datasets.

---

## Overview

The p-regions problem involves partitioning a set of spatial units into **p contiguous regions**.  
This library helps researchers and students generate **synthetic data** to test algorithms and benchmark solutions without requiring real-world datasets.

---

## Installation

Follow these steps to set up a local development environment.

1. **Clone the repository**
   
   ``` bash
   git clone https://github.com/DiegoHuerta1/Artificial-Instances-p-regions.git
   cd Artificial-Instances-p-regions
   ```

3. **Create and activate a virtual environment**

   
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```


4. **Upgrade pip and install dependencies**
   
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage


### Example: Generate Multiple Instances

```bash
python example_generate_folder.py
```

This script creates several instances and stores them in a specified folder.

---

## Repository Structure

- **Instance_Generator/** – Core Python module for creating artificial instances.  
- **example_generate_instance.py** – Example of how to produce a single instance.  
- **example_generate_folder.py** – Example of how to generate and save multiple instances.  
- **requirements.txt** – Python dependencies.  

---

## Output

The generated instance is returned as an `igraph.Graph` object with the following attributes:

**Node attributes**:
- `"x"` — list representing the node’s feature vector.
- `"name"` — string containing the node index.

**Graph attributes**:
- `"pos"` — dictionary mapping node IDs to their spatial coordinates.
- `"P"` — dictionary describing the partition of nodes.
- `"status"` — string indicating whether the partition `P` is optimal for the p-regions problem.


---

**Author:** Diego Huerta  






