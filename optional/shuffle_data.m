% load features and results
x = load('features_ordered.txt');
y = load('results_ordered.txt');

% shuffle matrice rows (shuffle merged x and y data)
xy = [x y];
xy = xy(randperm(size(xy, 1)), :);
x = xy(:, 1 : size(xy, 2) - 1)
y = xy(:, size(xy, 2) - 1)

% save shuffled features and results
save 'features.txt' x
save 'results.txt' y