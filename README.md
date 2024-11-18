# Thesis Repository

This repository contains the tools and resources used for conducting simulations and analyses related to the FMoW dataset. Due to its large size, the FMoW dataset is not included in this repository. Instead, instructions and tools for creating the dataset locally are provided in the `Dataset` folder.

---

## Steps to Get Started

### Step 1: Install Dependencies
Ensure that all required dependencies are installed. Run the following command in your terminal:

```bash

pip install -r requirements.txt

```

### Step 2: Explore the Dataset
Dataset's content can be explored using the `Exploration.ipynb` notebook. This notebook provides an overview of the dataset structure and key insights.

### Step 3: Simulate a Pipeline
To execute the desired pipeline, use the `Complete simulation.ipynb` notebook. This notebook includes all necessary functions, which are specified in its preamble, to facilitate pipeline execution and saves extracted features in the `features` folder, relevant images in `figures` folder and relevant data in `saved_data` folder.

### Step 4: Visualize Simulation Results
For visual representation of the pipeline's results, refer to the `Plot results.ipynb` notebook.

### Step 5: Compare Simulations
Compare the performance of different pipelines in terms of execution time and classification accuracy using the `Comparing results.ipynb` notebook. The required comparison functions are included in the notebook's preamble.

### Step 6: Retrieve Execution Times
Numerical execution times for various pipelines can be extracted using the `Restore times.py` script.

## Additional Notes

### Supporting Files
The repository includes additional Python files that are required to support the execution of the provided notebooks. Ensure that these files are kept in their respective locations for smooth operation.
