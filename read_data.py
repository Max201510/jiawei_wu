import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('learn_motivation.csv')
plt.scatter(data.motivation,data.score)
plt.xlabel('motivation')
plt.ylabel('score')
plt.show()
plt.scatter(data.motivation,data.self_test)
plt.xlabel('motivation')
plt.ylabel('self_test')
plt.show()