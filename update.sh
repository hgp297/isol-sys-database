#!/bin/bash
from=hgp297@stampede3.tacc.utexas.edu:/work2/05428/hgp297/stampede3/isol-sys-database/
to="./"
if [ $# -eq 1 ] && [ $1 = "t" ] 
then
    to=$from
    from="./"
    echo "Synchronizing from local to TACC!"
    rsync -zarvm --include="/src/" \
		--include="/resource/"\
		--include="/resource/ground_motions/" \
		--include="/resource/ground_motions/PEERNGARecords_Unscaled" \
		--include="/resource/loss/" \
		--include="isol_db_sbatch" \
		--include="/data/" \
		--include="*.AT2" \
		--include="*.g3" \
        --include="*.csv" \
        --include="update.sh" \
        --include="*.py" \
		--exclude="/src/main.py" \
        --exclude="*" \
		--exclude="/src/__pycache__/" \
		--exclude="/data/*.pickle" \
        "$from" "$to"
else
    echo "Synchronizing from tacc to local!"
    rsync -zarvm --include "/data/*.pickle" \
		--include="./isol_db_sbatch" \
		--include="./update.sh" \
        --include="/data/*.csv" \
		--include="/data/validation/" \
		--include="/data/loss/" \
		--include="/data/doe/" \
        --exclude="*" \
        "$from" "$to"
fi
