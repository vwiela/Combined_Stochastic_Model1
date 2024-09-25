# Combined_Stochastic_Model1

Code complimenting the project and paper "Combined Stochastic Model for the Evaluation of Cancer Progression and Patient Trajectories".

The src folder contains all the following code used for model building, optimization and visualization of the results:
- `Project.toml`and M̀anifest.toml` specifying the used packages in the Julia 1.10.4 environment.
- A script `functionalities.jl` containing useful functions for visualization.
- A script implementing the modified next reaction method algorithm used to simulate the combined process.
- One script for each model considered in the simulation study of the paper containing the corresponding likelihood functions.
- One script for running the multi-start optimization and one script per model for running the MCMC sampling.
- One script for benchmarking the times of using analytical or numerical likelihoods.
- One notebook for each model and experiment conducted in the simulation study. Exectuing the notebook will create or load the data used and visualize the results as seen in the paper.
- One notebook for the creation of additional figures used in the publication.

The data folder contains the datasets simulated and used for the experiments described in the notebooks.

The output folder contains all the results, e.g. from the parameter optimization or sampling.

A snapshot of the code, data and result files at the time of submition of the paper is available on Zenodo.\
Due to the large size the complete set of result files are only available on Zenodo or on request to the author.
