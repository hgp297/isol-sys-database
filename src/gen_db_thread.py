def generate_db(num, seed):
    from db import Database
    main_obj = Database(n_points=num, seed=seed, isol_wts=[1,2])
    main_obj.design_bearings(filter_designs=True)
    main_obj.design_structure(filter_designs=True)
    main_obj.scale_gms()
    output_dir = './outputs/seed_'+str(seed)+'_output/'
    main_obj.analyze_db('structural_db_seed_'+str(seed)+'.csv', save_interval=5,
                        output_path=output_dir)
    import pickle
    with open('../data/structural_db_seed_'+str(seed)+'.pickle', 'wb') as f:
        pickle.dump(main_obj, f)
        
import argparse

parser = argparse.ArgumentParser(
    description='Create db with seed, size, then run.')
parser.add_argument('size', metavar='N', type=int, nargs='?',
                    help='the number of points in the db object')
parser.add_argument('seed', metavar='s', type=int, nargs='?',
                    help='the seed that the db object will use')

args = parser.parse_args()
generate_db(args.size, args.seed)