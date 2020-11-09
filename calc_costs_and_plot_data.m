% load features (x) and results (y)
% note: data has to be shuffled, or have random values by default
x = load('features.txt');
y = load('results.txt');

% scale input features
x = scale_values(x);

% calculate the number of training set, cross validation and test set examples
x_rows = size(x, 1);
train_set_examples = x_rows * 0.6;
test_set_examples = x_rows * 0.2;

% the training set
x_train = x(round(1:train_set_examples), :);
y_train = y(round(1:train_set_examples), :);
% the cross validation set
x_cross = x(round(train_set_examples + 1) : round(train_set_examples + test_set_examples), :);
y_cross = y(round(train_set_examples + 1) : round(train_set_examples + test_set_examples), :);
% the test set
x_test = x(round(train_set_examples + test_set_examples + 1) : round(train_set_examples + test_set_examples + test_set_examples), :);
y_test = y(round(train_set_examples + test_set_examples + 1) : round(train_set_examples + test_set_examples + test_set_examples), :);

% set lambdas from 0 to 0.64
lambdas = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64];

disp('======================================');
disp('First Degree Polynomial');
disp('======================================');
% add a column for computing hypothesis
x_train_hypothesis = add_hypothesis_column(x_train);
x_cross_hypothesis = add_hypothesis_column(x_cross);
x_test_hypothesis = add_hypothesis_column(x_test);

% set initial theta for First Degree Polynomial hypothesis
theta = zeros(size(x_train_hypothesis, 2), 1);

% prepare lowest_cross_cost
lowest_cross_cost = inf;

% calculate costs for First Degree Polynomial hypothesis
for i=1:length(lambdas),
    lambda = lambdas(i);
    disp('================================');
    disp('Lambda:'), disp(lambda);
    disp('================================');
    disp('TRAINING SET');
    [theta_train, J_train, exit_flag] = gradient_descent(x_train_hypothesis, y_train, theta, lambda);
    J_train, exit_flag
    disp(''), disp('CROSS VALIDATION COST');
    J_cross = cost_function(x_cross_hypothesis, y_cross, theta_train, lambda)
    disp('');
    disp('TEST COST');
    J_test = cost_function(x_test_hypothesis, y_test, theta_train, lambda)
    disp('');

    if J_cross < lowest_cross_cost,
        lowest_cross_cost = J_cross;
        test_cost = J_test;
        test_theta = theta_train;
        train_x = x_train_hypothesis;
        cross_x = x_cross_hypothesis;
        test_x = x_test_hypothesis;
        test_hypothesis = 'First Degree Polynomial';
        test_lambda = lambda;
    end;
end;

disp('======================================');
disp('Second Degree Polynomial');
disp('======================================');
% extend features for Second Degree Polynomial hypothesis
x_train_polinomial2 = features_n_degree_polinomial(x_train, 2);
x_cross_polinomial2 = features_n_degree_polinomial(x_cross, 2);
x_test_polinomial2 = features_n_degree_polinomial(x_test, 2);

% add a column for computing hypothesis
x_train_polinomial2_hypothesis = add_hypothesis_column(x_train_polinomial2);
x_cross_polinomial2_hypothesis = add_hypothesis_column(x_cross_polinomial2);
x_test_polinomial2_hypothesis = add_hypothesis_column(x_test_polinomial2);

% set initial theta for Second Degree Polynomial hypothesis
theta = zeros(size(x_train_polinomial2_hypothesis, 2), 1);

% calculate costs for Second Degree Polynomial hypothesis
for i=1:length(lambdas),
    lambda = lambdas(i);
    disp('================================');
    disp('Lambda:'), disp(lambda);
    disp('================================');
    disp('TRAINING SET');
    [theta_train, J_train, exit_flag] = gradient_descent(x_train_polinomial2_hypothesis, y_train, theta, lambda);
    J_train, exit_flag
    disp(''), disp('CROSS VALIDATION COST');
    J_cross = cost_function(x_cross_polinomial2_hypothesis, y_cross, theta_train, lambda)
    disp('');
    disp('TEST COST');
    J_test = cost_function(x_test_polinomial2_hypothesis, y_test, theta_train, lambda)
    disp('');

    if J_cross < lowest_cross_cost,
        lowest_cross_cost = J_cross;
        test_cost = J_test;
        test_theta = theta_train;
        train_x = x_train_polinomial2_hypothesis;
        cross_x = x_cross_polinomial2_hypothesis;
        test_x = x_test_polinomial2_hypothesis;
        test_hypothesis = 'Second Degree Polynomial';
        test_lambda = lambda;
    end;
end;

disp('======================================');
disp('Second Degree Polynomial with all members');
disp('======================================');
% extend features for Second Degree Polynomial with all members hypothesis
x_train_polinomial2_all = features_2nd_degree_polinomial_all(x_train);
x_cross_polinomial2_all = features_2nd_degree_polinomial_all(x_cross);
x_test_polinomial2_all = features_2nd_degree_polinomial_all(x_test);

% add a column for computing hypothesis
x_train_polinomial2_all_hypothesis = add_hypothesis_column(x_train_polinomial2_all);
x_cross_polinomial2_all_hypothesis = add_hypothesis_column(x_cross_polinomial2_all);
x_test_polinomial2_all_hypothesis = add_hypothesis_column(x_test_polinomial2_all);

% set initial theta for Second Degree Polynomial with all members hypothesis
theta = zeros(size(x_train_polinomial2_all_hypothesis, 2), 1);

% calculate costs for Second Degree Polynomial with all members hypothesis
for i=1:length(lambdas),
    lambda = lambdas(i);
    disp('================================');
    disp('Lambda:'), disp(lambda);
    disp('================================');
    disp('TRAINING SET');
    [theta_train, J_train, exit_flag] = gradient_descent(x_train_polinomial2_all_hypothesis, y_train, theta, lambda);
    J_train, exit_flag
    disp(''), disp('CROSS VALIDATION COST');
    J_cross = cost_function(x_cross_polinomial2_all_hypothesis, y_cross, theta_train, lambda)
    disp('');
    disp('TEST COST');
    J_test = cost_function(x_test_polinomial2_all_hypothesis, y_test, theta_train, lambda)
    disp('');

    if J_cross < lowest_cross_cost,
        lowest_cross_cost = J_cross;
        test_cost = J_test;
        test_theta = theta_train;
        train_x = x_train_polinomial2_all_hypothesis;
        cross_x = x_cross_polinomial2_all_hypothesis;
        test_x = x_test_polinomial2_all_hypothesis;
        test_hypothesis = 'Second Degree Polynomial with all members';
        test_lambda = lambda;
    end;
end;

disp('======================================');
disp('Third Degree Polynomial');
disp('======================================');
% extend features for Third Degree Polynomial hypothesis
x_train_polinomial3 = features_n_degree_polinomial(x_train, 3);
x_cross_polinomial3 = features_n_degree_polinomial(x_cross, 3);
x_test_polinomial3 = features_n_degree_polinomial(x_test, 3);

% add a column for computing hypothesis
x_train_polinomial3_hypothesis = add_hypothesis_column(x_train_polinomial3);
x_cross_polinomial3_hypothesis = add_hypothesis_column(x_cross_polinomial3);
x_test_polinomial3_hypothesis = add_hypothesis_column(x_test_polinomial3);

% set initial theta for Third Degree Polynomial hypothesis
theta = zeros(size(x_train_polinomial3_hypothesis, 2), 1);

% calculate costs for Third Degree Polynomial hypothesis
for i=1:length(lambdas),
    lambda = lambdas(i);
    disp('================================');
    disp('Lambda:'), disp(lambda);
    disp('================================');
    disp('TRAINING SET');
    [theta_train, J_train, exit_flag] = gradient_descent(x_train_polinomial3_hypothesis, y_train, theta, lambda);
    J_train, exit_flag
    disp(''), disp('CROSS VALIDATION COST');
    J_cross = cost_function(x_cross_polinomial3_hypothesis, y_cross, theta_train, lambda)
    disp('');
    disp('TEST COST');
    J_test = cost_function(x_test_polinomial3_hypothesis, y_test, theta_train, lambda)
    disp('');

    if J_cross < lowest_cross_cost,
        lowest_cross_cost = J_cross;
        test_cost = J_test;
        test_theta = theta_train;
        train_x = x_train_polinomial3_hypothesis;
        cross_x = x_cross_polinomial3_hypothesis;
        test_x = x_test_polinomial3_hypothesis;
        test_hypothesis = 'Third Degree Polynomial';
        test_lambda = lambda;
    end;
end;

% set hypothesis
hypothesis = prediction = sigmoid(test_x * test_theta);
% set thresholds
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7];
% prepare highest_f1_score
highest_f1_score = 0;

% calculate precisions, recalls and f1_scores
% it -> index threshold
for it=1:length(thresholds),
    true_positives = true_negatives = false_positives = false_negatives = 0;

    % ih -> index hypothesis
    for ih=1:length(hypothesis),
        if hypothesis(ih) >= thresholds(it),
            prediction(ih) = 1;
        else
            prediction(ih) = 0;
        end;

        if prediction(ih) == y_test(ih) && prediction(ih) == 1,
            true_positives = true_positives + 1;
        elseif prediction(ih) == y_test(ih) && prediction(ih) == 0,
            true_negatives = true_negatives + 1;
        elseif prediction(ih) ~= y_test(ih) && prediction(ih) == 1,
            false_positives = false_positives + 1;
        else
            false_negatives = false_negatives + 1;
        end;
    end;

    disp('================================');
    disp('Threshold:'), disp(thresholds(it));
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    misclassification_error = (false_positives + false_negatives) / length(y_test)

    if f1_score > highest_f1_score,
        highest_f1_score = f1_score;
        optimal_threshold = thresholds(it);
        optimal_precision = precision;
        optimal_recall = recall;
        optimal_misclassification_error = misclassification_error;
    end;
end;

% display lowest_cross_cost
disp(''), disp('======================================');
disp('LOWEST CROSS COST: '), disp(lowest_cross_cost);
disp('TEST COST: '), disp(test_cost);
disp('TEST THETA: '), disp(test_theta);
disp('TEST HYPOTHESIS: '), disp(test_hypothesis);
disp('TEST LAMBDA: '), disp(test_lambda);
disp('======================================');

% display highest_f1_score
disp('HIGHEST F1 SCORE: '), disp(highest_f1_score);
disp('THRESHOLD: '), disp(optimal_threshold);
disp('PRECISION: '), disp(optimal_precision);
disp('RECALL: '), disp(optimal_recall);
disp('MISCLASSIFICATION ERROR: '), disp(optimal_misclassification_error);
disp('======================================');

[error_train, error_cross] = learning_curve(train_x, y_train, cross_x, y_cross, test_lambda);

% plot blue curve (3 thickness) to showcase Train error
plot(error_train, ':', 'linewidth', 3);
hold on;
% plot brown curve (3 thickness) to showcase Cross validation error
plot(error_cross, '-.', 'linewidth', 3);

xlbl = xlabel('Number of training examples');
ylbl = ylabel('Errors');
lgnd = legend('Train error curve', 'Cross validation error curve');
set([xlbl, ylbl, lgnd], 'fontsize', 16);
% remove border
box off;

% print learning curve graph
print -dpng 'LearningCurves.png'