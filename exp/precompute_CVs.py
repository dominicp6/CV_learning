from Experiment import Experiment

exp = Experiment(
    location="/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/desres/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein",
    features='dihedrals')

exp.implied_timescale_analysis(max_lag='200ns', increment=5)
#exp.compute_cv('TICA', lagtime=)
#exp.free_energy_plot(features=['TICA0', 'TICA1'], feature_nicknames=['TICA0', 'TICA1'])
