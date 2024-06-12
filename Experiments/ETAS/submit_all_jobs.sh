#!/bin/bash

qsub -v CONFIG=invert_etas_Japan_25 job.sh;
qsub -v CONFIG=invert_etas_incomplete_California_25 job.sh;
qsub -v CONFIG=invert_etas_synthetic_California_25 job.sh;
qsub -v CONFIG=invert_etas_SaltonSea_10 job.sh;
qsub -v CONFIG=invert_etas_SanJac_10 job.sh;
qsub -v CONFIG=invert_etas_WHITE_06 job.sh;
qsub -v CONFIG=invert_etas_SCEDC_20 job.sh;
qsub -v CONFIG=invert_etas_SCEDC_25 job.sh;
qsub -v CONFIG=invert_etas_SCEDC_30 job.sh;
qsub -v CONFIG=invert_etas_ComCat_25 job.sh;