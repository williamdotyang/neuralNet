file_targets = fopen('targets','r');
file_outputs = fopen('outputs','r');
targets = fscanf(file_targets, '%f');
outputs = fscanf(file_outputs, '%f');
plotroc(targets',outputs')