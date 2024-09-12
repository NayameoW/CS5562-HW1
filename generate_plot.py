import matplotlib.pyplot as plt

# 数据
eps_values = [2/255, 4/255, 6/255, 8/255, 10/255]
adversarial_accuracy = [38.15, 15.05, 5.7, 2.85, 2.1]

plt.figure(figsize=(10, 6))
plt.plot(eps_values, adversarial_accuracy, 'r-o', label='Adversarial Images')

plt.title('Model accuracy on various epsilons\n Batch number 20, Batch size 100, Alpha 2/255', fontsize=16)
plt.xlabel('Epsilon (eps)', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)

plt.ylim(0, 50)

plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(fontsize=12)

for i, txt in enumerate(adversarial_accuracy):
    plt.annotate(f'{txt}%', (eps_values[i], txt), textcoords="offset points", xytext=(0,-15), ha='center')

plt.tight_layout()
plt.show()