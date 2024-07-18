# isol-sys-database

A database of isolated steel frames and their performance under earthquakes.

## Description

This repository is aimed at generating and analyzing isolated steel frames for their performance.
Most of the repository is dedicated towards automating the design of steel moment and braced frames isolated with friction or lead rubber bearings.
The designs are then automatically constructed in OpenSeesPy and subjected to a full nonlinear time history analysis.
The database is revolved around generating series of structures spanning the range of a few random design variables dictating over/under design of strength and displacement capacity variables, along with a few isolator design parameters.

Decision variable prediction via the SimCenter toolbox is also available (work in progress).

An analysis folder is available with some scripts performing data visualization and machine learning predictions.
The database is utilized to generate inverse design targeting specific structural performance.

## Dependencies

A comprehensive ```.yaml``` file containing the below dependencies is available for virtual environment setup (in Conda). However, it is derived directly from my working environment and includes some personal software, such as the Spyder IDE. Please remove these as necessary.

* Structural software:
	* OpenSeesPy 3.4.0
	* Python 3.9

* Data structure management:
	* Pandas 2.2.0+
	* Numpy 1.22.4+
	* Scipy 1.12.0+

* Machine learning analyses (required for design of experiment, inverse design):
	* Scikit-learn

* Visualization:
	* Matplotlib
	* Seaborn

* Decision-variable prediction:
	* Pelicun 3.1+

## Setup

Prepare a directory for each individual run's outputs under ```src/outputs/```, as well as the data output directory under ```data```. This is not done automatically in this repository since these directories should change if parallel running is desired.

Once that is completed, with the exception of post-experiment analyses, such as making plots and prediction models, the repository relies on relative pathing and should be able to run out-of-the-box. For visualizations and analyses from the ```src/analyses/``` folder, ensure that all ```sys.path.insert``` calls reference the directory with the files that generate ML models (particularly ```doe.py```). Additionally, ensure that all pickled and CSV objects reference within are pointed to a proper data file (which are not provided in this GitHub repository, but are available in the DesignSafe Data Depot upload).

## Usage
The database generation is handled through main_\* scripts available in the ```src``` folder.
```src/analyses/``` contains scripts for data visualization and results processing.

Files under ```src/``` titled gen_\* and val_\* are written for HPC-utilizing parallel computations and are not detailed here.

### Generating an initial database

An initial database of size `n_pts`, distributed randomly uniform via Latin Hypercube sampling, can be generated with 

    from db import Database
    db_obj = Database(n_pts, seed=123)
    
The database can be limited to just certain systems using the `struct_sys_list` and `isol_sys_list` arguments, defaulting to generate both moment frames and concentric braced frames, both friction and lead rubber bearings. This database holds all methods for further design and analyses and therefore must be generated. Then, generate designs for the current database of dimensionless design parameters.

    db_obj.design_bearings(filter_designs=True)
    db_obj.design_structure(filter_designs=True)
    
A full list of unfiltered designs is available in `db_obj.generated_designs`. After removing for unreasonable designs, there should be `n_pts` designs remaining, stored in `db_obj.retained_designs`. To prepare the ground motions for the analyses, performance

    db_obj.scale_gms()
    
which requires and augments the `retained_designs` attribute. Finally, perform the nonlinear dynamic analyses with

    db_obj.analyze_db('my_database.csv')
    
It is then recommended to store the data in a pickle file as well to preserve data structures in drift/velocity/acceleration outputs.

    import pickle
    with open('../data/my_run.pickle', 'wb') as f:
        pickle.dump(db_obj, f)
    
### Analyzing individual runs

Some troubleshooting tools are provided. From any row of the design DataFrames, a Building object can be created. For example, a diagnostic run is provided below.

    from building import Building
    bldg = Building(run)
    bldg.model_frame()
    bldg.apply_grav_load()
    T_1 = bldg.run_eigen()
    bldg.provide_damping(80, method='SP', zeta=[0.05], modes=[1])
    dt = 0.005
    ok = bldg.run_ground_motion(run.gm_selected, 
                            run.scale_factor*1.0, 
                            dt, T_end=60.0)

    from plot_structure import plot_dynamic
    plot_dynamic(run)
    
### Performing design-of-experiment

This task requires for a Database object to exist, and that results from analyses already exist for this (stored in `db_obj.ops_analysis`). To load an existing pickled Database object,

    import pickle
    with open('my_run.pickle', 'rb') as f:
        db_obj = pickle.load(f)

Then, first calculate collapse probabilities (since DoE is tied to targeting collapse probability).

    db_obj.calculate_collapse()
    db_obj.perform_doe(n_set=200,batch_size=5, max_iters=1500, strategy='balanced')
    
`n_set` determines how many points to build the ML object for DoE from. `batch_size` and `max_iters` are run controls for size of each DoE batch and maximum number of points added before exhaustion. `strategy` specifies the DoE strategy, which can be `explore` to target model variance, `exploit` to target model bias, or `balanced` to weigh both equally.


### Running loss analysis with Pelicun

Assuming that a database is available in the Database object's `ops_analysis` attribute (or `doe_analysis`), damage and loss can be calculated using 

    db_obj.run_pelicun(db_obj.ops_analysis, collect_IDA=False,
                    cmp_dir='../resource/loss/')

    import pickle
    loss_path = '../data/loss/'
    with open(loss_path+'my_damage_loss.pickle', 'wb') as f:
        pickle.dump(db_obj, f)
        
To calculate theoretical maximum damage/loss of the building, run

    db_obj.calc_cmp_max(db_obj.ops_analysis,
                cmp_dir='../resource/loss/')

These are stored in the Database object's `loss_data` and `max_loss` attributes, respectively.

### Validating a design in incremental dynamic analysis

This assumes that a Database object exists. Specify the design of the validated design using a dictionary.

    sample_dict = {
        'gap_ratio' : 0.6,
        'RI' : 2.25,
        'T_ratio': 2.16,
        'zeta_e': 0.25
    }
    
Then, prepare an IDA of three MCE levels (1.0, 1.5, 2.0x by default), perform the IDA, and store the results.

    design_df = pd.DataFrame(sample_dict, index=[0])
    db_obj.prepare_ida_legacy(design_df)
    db_obj.analyze_ida('ida_sample.csv')

    import pickle
    with open(validation_path+'my_ida.pickle', 'wb') as f:
        pickle.dump(db_obj, f)

## Interpreting results

A list of variables generated in the `ops_analysis` and `doe_analysis` object is available in ```resource/variable_list.xlsx```.

## TODO/Personal notes:
A reminder that this database is dependent on the OpenSees compatible with Python=3.9.
See opensees_build/locations/ for location of a working Opensees.pyd code.

## Research tools utilized

* [OpenSeesPy](https://github.com/zhuminjie/OpenSeesPy)
* [SimCenter Pelicun](https://github.com/NHERI-SimCenter/pelicun)