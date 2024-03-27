import argparse
import os
import pandas as pd


def main(args):
    # find all csv files in args.cosine_sim_dump_dir
    csv_files = [f for f in os.listdir(args.cosine_sim_dump_dir) if f.endswith('.csv')]

    # iterate over all csv files in args.cosine_sim_dump_dir
    # load them using pandas
    # merge them into a single dataframe object
    all_results = pd.DataFrame()
    for f in csv_files:
        f_path = os.path.join(args.cosine_sim_dump_dir, f)
        df = pd.read_csv(f_path)
        all_results = pd.concat([all_results, df])
    
    # print summary statistics
    print('######### SUMMARY #########')
    print(f'average cosine similarity: {all_results["cosine_similarity"].mean()}')
    print('##########################')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="Collect saliency cosine sim results for summary")
    parser.add_argument("--cosine_sim_dump_dir", type=str, help="src dir of dumped cosine sim results", required=True)
    args = parser.parse_args()
    main(args)  # Call the main function