1. First, download dataset and transform them to the 'Standard Format' with the codes in './Data_Preprocessing/', further instructions see readme.txt in the folders.
	- As a result of this step, each dataset will become a set of narrow format .csv files stored in a Folder named 'Dataset_name/Dataset_Separation/'

2. Put datasets in standard format from step 1 into existing folders in './GNN_INTP/Datasets/'
	- When set correctly, the path will be like './GNN_INTP/Datasets/ABO_res250/Dataset_Separation/'

3. Run 'convert_to_region4.py' in './GNN_INTP/Datasets/', this will further preprocessing the data
	- You will get 'ABO_res250_reg4c' and 'SAQN_res250_reg4c'
	- Copy json files in './GNN_INTP/Datasets/metas/' into those folders according to the file name, and rename them all as 'meta_data.json'
	- When set correctly, 'ABO_res250_reg4c' and 'SAQN_res250_reg4c' folders will have following folders and files:
		- A 'Dataset_Separation' folder, which saves dataset .csv files
		- A 'meta_data.json' file, which saves meta data of the dataset
		- A 'log.pkl' file, which saves how different op_names are seperated in different leave-one-area-out fold

4. Run '02.Fold_Divide.py' in './GNN_INTP/Datasets/', this will analyse informations for different folds of leave-one-area-out cross validation
	- When set correctly, 'ABO_res250_reg4c' and 'SAQN_res250_reg4c' folders will add following folder:
		- A 'Folds_info' folder, which saves information about each fold of 4-fold leave-one-area-out cross validation

5. Run Overall Experiments & Ablation study (without LSPE, LSPE will be introduced separetly in 6):
	- 1. Set Experiments configs in './GNN_INTP/configs_files/config_kfold_trans.py'
		We use experiment management software W&B, you need to fill in your own api_key, entity_name, e_mail, and conda_env.
		You also need to set sweep_config to define which models on which datasets you want to carry out the experiment.
		The naming in the code is slightly different as they are in the paper:
			GCN - GCN
			GAT - GAT
			GSAGE - GraphSAGE
			KSAGE - KCN (using GraphSAGE as backbone)
			PEGSAGE - PE-GNN (using GraphSAGE as backbone)
			NODEFORMER - NodeFormer
			LSPE - GNN-LSPE
			FEGSAGE_CA - ISSI
			FEGSAGE_CRA - ISSI w/o TR
			FEGSAGE_CCA - ISSI w/o SS
			FEGSAGE_CNA - CESI Null
	- 2. Set use_config = "config_kfold_trans" in './GNN_INTP/config_manager.py'
	- 3. Run './GNN_INTP/01.wandb_start.py'
		We use a slurm based HPC, the task is triggered in line 154 in './GNN_INTP/slurm_wrapper.py' by submitting generated sbatch file to slurm. If you use other running environments, please edit this part by yourself

6. Run GNN-LSPE Experiments
	GNN-LSPE uses a very different way of implementation, so we decide not to merge it into our standard working process to make sure we are not making any mistakes
	- 1. First, we need to prepare LSPE datasets, run './GNN_INTP/make_graphs.py' to do this
		We use a slurm based HPC, the task is triggered in line 76 in './GNN_INTP/make_graphs.py' by submitting generated sbatch file to slurm. If you use other running environments, please edit this part by yourself
		After all datasets sucessfully generated, compress all folders in $WSDIR/{dataset}/pkl/ as .tgz files one by one
	- 2. Set Experiments configs in './GNN_INTP/configs_files/config_kfold_lspe.py'
	- 3. Set use_config = "config_kfold_lspe" in './GNN_INTP/config_manager.py'
	- 4. Run './GNN_INTP/01.wandb_start.py'

