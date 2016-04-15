figure
x1 = [25 50 75 100];
y1 = [0.76103896103896107 0.76627705627705622 0.77080086580086571 0.7853246753246752];
x2 = [5 10 15 20 25];
y2 = [0.76940010263890013 0.76627705627705622 0.74984126984126964 0.76914141414141413 0.7952380952380953];
plot(x1, y1, '-o')
title('Accuracy over different number of epochs')
xlabel('number of epochs')
ylabel('accuracy')

figure
plot(x2, y2, '-o')
title('Accuracy over different number of folds')
xlabel('number of folds')
ylabel('accuracy')

figure
file_targets = fopen('targets','r');
file_outputs = fopen('outputs','r');
targets = fscanf(file_targets, '%f');
outputs = fscanf(file_outputs, '%f');
plotroc(targets',outputs')
