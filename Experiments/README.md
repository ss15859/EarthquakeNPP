# Running the Experiments

We ran all the NPP models on [Isambard Phase 3](https://gw4-isambard.github.io/docs/), using a node with Nvidia Ampere GPU with 4x Nvidia A100 40GB SXM “Ampere” GPUs and AMD EPYC 7543P 32-Core Processor “Milan” CPU using torch==1.12.0 and cuda==11.3. Isambard is a HPC service provided by [GW4](http://gw4.ac.uk/) and the [UK Met Office](https://www.metoffice.gov.uk/). The system is funded by [EPSRC](http://www.epsrc.ac.uk/) and is one of a number of [Tier-2 HPC facilities](http://www.hpc-uk.ac.uk/facilities/) in the UK.

## NSTPP 

Chen et al. [(2020)]((https://arxiv.org/pdf/2011.04583)) split the data into windows based on a fixed length interval (1 month). We found that pre-specifying a time window like this led to sequences that were either too long or consisted of many sequences that contain a single event. For our windowing, we found the time interval $[0,t_1]$ such that the minimum length of sequence was 3. We found that this approach led to 'Events per sequence' histograms that better resembled those in Chen et al. [(2020)]((https://arxiv.org/pdf/2011.04583)) Figure 11. This is implemented in the script `create_dataset.py`.

Furthermore, for Chen et al. [(2020)]((https://arxiv.org/pdf/2011.04583)) 's implementation of NSTPP: "spatial variables have been standardized using the empirical mean and standard deviation from the training set". In order for all log-likelihood values to be scaled equivalently,  $\log(\det(\Sigma))$ has to be subtracted from the spatial component of the log-likelihood for NSTPP, where $\Sigma$ is the standard deviation matrix of the training set.
