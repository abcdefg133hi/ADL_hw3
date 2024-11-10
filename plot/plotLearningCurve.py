import json
import os
import numpy as np
import matplotlib.pyplot as plt

iteration = [1,2,3,4,5]
with open("result.json", 'r') as fin:
    data = json.load(fin)

plt.plot(iteration, data, label='public perplexity', marker='o')
plt.title("Public Learning Curve")
plt.xlabel("Training Epochs")
plt.ylabel("Perplexity")
plt.legend()
plt.savefig("LearningCurve.png")
#plt.show()


