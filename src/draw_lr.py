import matplotlib.pyplot as plt

num_epochs = 50

acc_3 = []
acc_4 = []
acc_5 = []
acc_6 = []
epochs = [i+1 for i in range(num_epochs)]

with open('./txt/lr_acc_resnet18.txt', 'r') as f:
    for _ in range(num_epochs):
        line = f.readline()
        # 将每行内容按空格分割成四个元素
        items = line.strip().split()
        # 将每个元素转换为相应的数据类型，并添加到对应的列表中
        acc_3.append(float(items[0]))
        acc_4.append(float(items[1]))
        acc_5.append(float(items[2]))
        acc_6.append(float(items[3]))

parameters = {
    'axes.labelsize': 20,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 1,
    'figure.figsize': [12.0, 5.0],
    'font.sans-serif': 'Arial'
}
plt.rcParams.update(parameters)

plt.plot(epochs, acc_3, label='lr=1e-3')
plt.plot(epochs, acc_4, label='lr=1e-4')
plt.plot(epochs, acc_5, label='lr=1e-5')
plt.plot(epochs, acc_6, label='lr=1e-6')


plt.legend()
plt.title('Vit')
plt.xlabel('Epochs')
plt.ylabel('Test Acc')
plt.savefig('fig/lr_resnet18.png', bbox_inches='tight')
plt.savefig('fig/lr_resnet18.pdf', bbox_inches='tight')
plt.show()
