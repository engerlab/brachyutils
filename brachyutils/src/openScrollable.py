import pickle
fig_name = "/home/majd/Software/tg186-validation/doseGroundTruth.pickle"
fig = pickle.load(open(fig_name, 'rb'))
fig.show()