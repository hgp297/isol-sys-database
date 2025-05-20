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
		--include="/src/inputs/"\
		--include="/resource/ground_motions/" \
		--include="/resource/ground_motions/PEERNGARecords_Unscaled" \
		--include="/resource/loss/" \
		--include="*.AT2" \
        --include="*.csv" \
        --include="*.py" \
		--include="*.in" \
		--include="*.cfg" \
		--exclude="/src/main.py" \
        --exclude="*" \
		--exclude="/src/__pycache__/" \
		--exclude="/data/*.pickle" \
        "$from" "$to"
else
	to="/mnt/c/Users/hgp/Documents/bezerkeley/research/isol-sys-database/taccdata/"
    echo "Synchronizing from tacc to local!"
    rsync -zarvm --include "/" \
		--include "/data/" \
		--include "/data/initial/***" \
		--include="/data/validation/***" \
		--exclude="*" \
        "$from" "$to"
fi
