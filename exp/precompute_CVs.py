from Experiment import Experiment

exp_list = []
CVs = {'PCA': None, 'TICA': None, 'DMD': None, 'VAMP': None}

for exp in exp_list:
    print(f"Doing {exp}.")
    # load exp
    experiment = Experiment(location=exp, )

    # compute CVs
    for CV in CVs.items():
        print(f"- Doing {CV[0]}")
        # do something
        # save CV to disk