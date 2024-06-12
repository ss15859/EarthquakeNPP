#!/bin/bash

qsub -v CONFIG=Japan_25 job.sh;
qsub -v CONFIG=incomplete_California_25 job.sh;
qsub -v CONFIG=synthetic_California_25 job.sh;
qsub -v CONFIG=SaltonSea_10 job.sh;
qsub -v CONFIG=SanJac_10 job.sh;
qsub -v CONFIG=WHITE_06 job.sh;
qsub -v CONFIG=SCEDC_20 job.sh;
qsub -v CONFIG=SCEDC_25 job.sh;
qsub -v CONFIG=SCEDC_30 job.sh;
qsub -v CONFIG=ComCat_25 job.sh;